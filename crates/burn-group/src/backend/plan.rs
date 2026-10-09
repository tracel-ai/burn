use std::collections::HashMap;

use burn_backend::{DType, Slice};
use burn_ir::{
    ActivationOperationIr, BaseOperationIr, BinaryOpIr, BoolOperationIr, CatOpIr,
    EmbeddingBackwardOpIr, EmbeddingOpIr, FlipOpIr, FloatOperationIr, IntOperationIr,
    LinearBiasBackwardOpIr, LinearOpIr, LinearWeightBackwardOpIr, LinearXBackwardOpIr, MatmulOpIr,
    ModuleOperationIr, NumericOperationIr, OperationIr, ScalarOpIr, ShapeOpIr, SliceOpIr, TensorId,
    TensorIr, UnaryOpIr,
};

use crate::{
    DimSplit, EmbeddingBackwardRule, EmbeddingRule, GroupPlacement, LinearRule, Linearity,
    MatmulRule, OpPlacement, PlacementShapes, ReduceRule, Reduction, ReshapeRule, WholeDimRule,
};

/// Where each tensor of one op must be for the op to run on every member, where its outputs land,
/// and what each member runs.
#[derive(Debug)]
pub struct OpPlan {
    inputs: HashMap<TensorId, GroupPlacement>,
    outputs: HashMap<TensorId, GroupPlacement>,
    execution: Execution,
}

/// What each member runs, when it is not simply the op on its own shards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Execution {
    EveryMember,
    /// Member 0 runs the op and the others take a copy, so random values agree.
    FirstMemberCopied,
    /// Only member 0 adds the bias: the output is a partial sum, which must count it once.
    BiasOnFirstMember,
    /// A mean over the split dim: each member sums its chunk and divides by the whole count.
    SumThenDivide {
        count: usize,
    },
    /// A lookup in weights split by vocab row, each member zeroing the rows it does not hold.
    VocabLookup {
        split: DimSplit,
    },
    /// The gradient of weights split by vocab row, each member keeping its own tokens' gradient.
    VocabBackward {
        split: DimSplit,
    },
}

impl OpPlan {
    /// An op without a rule, or whose inputs would need two placements at once, gathers every
    /// input and runs whole on every member.
    pub fn new(
        op: &OperationIr,
        placements: &HashMap<TensorId, GroupPlacement>,
        members: usize,
    ) -> Self {
        Analysis {
            placements,
            members,
        }
        .plan(op)
        .unwrap_or_else(|| Self::gathered(op))
    }

    pub fn input(&self, id: &TensorId) -> GroupPlacement {
        self.inputs[id]
    }

    pub fn outputs(&self) -> &HashMap<TensorId, GroupPlacement> {
        &self.outputs
    }

    pub fn execution(&self) -> Execution {
        self.execution
    }

    /// A slice that takes a split dim whole takes each member's whole chunk, which the global
    /// range would overrun.
    pub fn take_split_dims_whole(&self, op: &mut OperationIr) {
        let (OperationIr::BaseFloat(BaseOperationIr::Slice(desc))
        | OperationIr::BaseInt(BaseOperationIr::Slice(desc))
        | OperationIr::BaseBool(BaseOperationIr::Slice(desc))) = op
        else {
            return;
        };
        if let GroupPlacement::Sharded { dim } = self.input(&desc.tensor.id)
            && let Some(range) = desc.ranges.get_mut(dim)
        {
            *range = Slice::new(0, None, 1);
        }
    }

    fn gathered(op: &OperationIr) -> Self {
        let replicated = |tensor: &TensorIr| (tensor.id, GroupPlacement::Replicated);
        Self {
            inputs: op.inputs().map(replicated).collect(),
            outputs: op.outputs().map(replicated).collect(),
            execution: Execution::EveryMember,
        }
    }
}

struct Analysis<'a> {
    placements: &'a HashMap<TensorId, GroupPlacement>,
    members: usize,
}

impl Analysis<'_> {
    fn plan(&self, op: &OperationIr) -> Option<OpPlan> {
        match op {
            OperationIr::BaseFloat(op) | OperationIr::BaseInt(op) | OperationIr::BaseBool(op) => {
                self.base(op)
            }
            OperationIr::NumericFloat(dtype, op) | OperationIr::NumericInt(dtype, op) => {
                self.numeric(*dtype, op)
            }
            OperationIr::Float(_, op) => self.float(op),
            OperationIr::Int(op) => self.int(op),
            OperationIr::Bool(op) => self.bool(op),
            OperationIr::Module(op) => self.module(op),
            OperationIr::Activation(op) => self.activation(op),
            OperationIr::Init(_)
            | OperationIr::Drop(_)
            | OperationIr::Custom(_)
            | OperationIr::Distributed(_) => None,
        }
    }

    fn base(&self, op: &BaseOperationIr) -> Option<OpPlan> {
        match op {
            BaseOperationIr::Reshape(desc) => self.reshape(desc),
            BaseOperationIr::SwapDims(desc) => {
                let mut axes: Vec<usize> = (0..desc.input.shape.num_dims()).collect();
                axes.swap(desc.dim1, desc.dim2);
                self.permute(&desc.input, &desc.out, &axes)
            }
            BaseOperationIr::Permute(desc) => self.permute(&desc.input, &desc.out, &desc.axes),
            BaseOperationIr::Flip(desc) => self.flip(desc),
            BaseOperationIr::Expand(desc) => self.expand(desc),
            BaseOperationIr::Slice(desc) => self.slice(desc),
            BaseOperationIr::Cat(desc) => self.cat(desc),
            BaseOperationIr::MaskWhere(desc) => self.elementwise(
                Linearity::Nonlinear,
                [&desc.tensor, &desc.mask, &desc.value],
                &desc.out,
            ),
            BaseOperationIr::MaskFill(desc) => {
                self.elementwise(Linearity::Nonlinear, [&desc.tensor, &desc.mask], &desc.out)
            }
            BaseOperationIr::Equal(desc) | BaseOperationIr::NotEqual(desc) => {
                self.binary(Linearity::Nonlinear, desc)
            }
            BaseOperationIr::EqualElem(desc) | BaseOperationIr::NotEqualElem(desc) => {
                self.scalar(Linearity::Nonlinear, desc)
            }
            BaseOperationIr::Cast(desc) => {
                self.elementwise(Linearity::Nonlinear, [&desc.input], &desc.out)
            }
            BaseOperationIr::Empty(desc)
            | BaseOperationIr::Ones(desc)
            | BaseOperationIr::Zeros(desc) => self.created(&desc.out),
            BaseOperationIr::All(desc) | BaseOperationIr::Any(desc) => {
                self.whole_dims(&desc.input, &desc.out, Reduction::All)
            }
            BaseOperationIr::AllDim(desc) | BaseOperationIr::AnyDim(desc) => {
                self.whole_dims(&desc.input, &desc.out, Reduction::Dim(desc.axis))
            }
            BaseOperationIr::Unfold(_)
            | BaseOperationIr::SliceAssign(_)
            | BaseOperationIr::Select(_)
            | BaseOperationIr::SelectAssign(_)
            | BaseOperationIr::Gather(_)
            | BaseOperationIr::Scatter(_)
            | BaseOperationIr::ScatterNd(_)
            | BaseOperationIr::GatherNd(_)
            | BaseOperationIr::RepeatDim(_) => None,
        }
    }

    fn numeric(&self, dtype: DType, op: &NumericOperationIr) -> Option<OpPlan> {
        match op {
            NumericOperationIr::Add(desc) | NumericOperationIr::Sub(desc) => {
                self.binary(Linearity::Linear, desc)
            }
            NumericOperationIr::Mul(desc) => self.binary(Linearity::Multilinear, desc),
            NumericOperationIr::Div(desc) => self.binary(Linearity::div(dtype), desc),
            NumericOperationIr::Rem(desc)
            | NumericOperationIr::Powi(desc)
            | NumericOperationIr::Greater(desc)
            | NumericOperationIr::GreaterEqual(desc)
            | NumericOperationIr::Lower(desc)
            | NumericOperationIr::LowerEqual(desc) => self.binary(Linearity::Nonlinear, desc),
            NumericOperationIr::MulScalar(desc) => self.scalar(Linearity::Linear, desc),
            NumericOperationIr::DivScalar(desc) => self.scalar(Linearity::div_scalar(dtype), desc),
            NumericOperationIr::AddScalar(desc)
            | NumericOperationIr::SubScalar(desc)
            | NumericOperationIr::RemScalar(desc)
            | NumericOperationIr::PowiScalar(desc)
            | NumericOperationIr::GreaterElem(desc)
            | NumericOperationIr::GreaterEqualElem(desc)
            | NumericOperationIr::LowerElem(desc)
            | NumericOperationIr::LowerEqualElem(desc)
            | NumericOperationIr::ClampMin(desc)
            | NumericOperationIr::ClampMax(desc) => self.scalar(Linearity::Nonlinear, desc),
            NumericOperationIr::Neg(desc) => self.unary(Linearity::Linear, desc),
            NumericOperationIr::Abs(desc) | NumericOperationIr::Sign(desc) => {
                self.unary(Linearity::Nonlinear, desc)
            }
            NumericOperationIr::Clamp(desc) => {
                self.elementwise(Linearity::Nonlinear, [&desc.tensor], &desc.out)
            }
            NumericOperationIr::Full(desc) => self.created(&desc.out),
            NumericOperationIr::IntRandom(desc) => self.random(&desc.out),
            NumericOperationIr::Sum(desc) => self.sum(&desc.input, &desc.out, Reduction::All),
            NumericOperationIr::SumDim(desc) => {
                self.sum(&desc.input, &desc.out, Reduction::Dim(desc.axis))
            }
            NumericOperationIr::Mean(desc) => {
                self.mean(dtype, &desc.input, &desc.out, Reduction::All)
            }
            NumericOperationIr::MeanDim(desc) => {
                self.mean(dtype, &desc.input, &desc.out, Reduction::Dim(desc.axis))
            }
            NumericOperationIr::Prod(desc)
            | NumericOperationIr::Max(desc)
            | NumericOperationIr::Min(desc)
            | NumericOperationIr::MaxAbs(desc) => {
                self.whole_dims(&desc.input, &desc.out, Reduction::All)
            }
            NumericOperationIr::ProdDim(desc)
            | NumericOperationIr::MaxDim(desc)
            | NumericOperationIr::MinDim(desc)
            | NumericOperationIr::ArgMax(desc)
            | NumericOperationIr::ArgMin(desc)
            | NumericOperationIr::MaxAbsDim(desc)
            | NumericOperationIr::TopK(desc)
            | NumericOperationIr::ArgTopK(desc) => {
                self.whole_dims(&desc.input, &desc.out, Reduction::Dim(desc.axis))
            }
            NumericOperationIr::CumSum(desc)
            | NumericOperationIr::CumProd(desc)
            | NumericOperationIr::CumMin(desc)
            | NumericOperationIr::CumMax(desc) => {
                self.whole_dims(&desc.input, &desc.out, Reduction::Dim(desc.axis))
            }
            NumericOperationIr::SumDims(_)
            | NumericOperationIr::MaxDimWithIndices(_)
            | NumericOperationIr::MinDimWithIndices(_)
            | NumericOperationIr::TopKWithIndices(_)
            | NumericOperationIr::Sort(_)
            | NumericOperationIr::SortWithIndices(_)
            | NumericOperationIr::ArgSort(_)
            | NumericOperationIr::Pad(_) => None,
        }
    }

    fn float(&self, op: &FloatOperationIr) -> Option<OpPlan> {
        match op {
            FloatOperationIr::Exp(desc)
            | FloatOperationIr::Log(desc)
            | FloatOperationIr::Log1p(desc)
            | FloatOperationIr::Erf(desc)
            | FloatOperationIr::Sqrt(desc)
            | FloatOperationIr::Cos(desc)
            | FloatOperationIr::Cosh(desc)
            | FloatOperationIr::Sin(desc)
            | FloatOperationIr::Sinh(desc)
            | FloatOperationIr::Tan(desc)
            | FloatOperationIr::Tanh(desc)
            | FloatOperationIr::ArcCos(desc)
            | FloatOperationIr::ArcCosh(desc)
            | FloatOperationIr::ArcSin(desc)
            | FloatOperationIr::ArcSinh(desc)
            | FloatOperationIr::ArcTan(desc)
            | FloatOperationIr::ArcTanh(desc)
            | FloatOperationIr::Round(desc)
            | FloatOperationIr::Floor(desc)
            | FloatOperationIr::Ceil(desc)
            | FloatOperationIr::Trunc(desc)
            | FloatOperationIr::Recip(desc)
            | FloatOperationIr::IsNan(desc)
            | FloatOperationIr::IsInf(desc) => self.unary(Linearity::Nonlinear, desc),
            FloatOperationIr::PowfScalar(desc) => self.scalar(Linearity::Nonlinear, desc),
            FloatOperationIr::ArcTan2(desc)
            | FloatOperationIr::Powf(desc)
            | FloatOperationIr::Hypot(desc) => self.binary(Linearity::Nonlinear, desc),
            FloatOperationIr::IntoInt(desc) => {
                self.elementwise(Linearity::Nonlinear, [&desc.input], &desc.out)
            }
            FloatOperationIr::Matmul(desc) => self.matmul(desc),
            FloatOperationIr::Random(desc) => self.random(&desc.out),
            FloatOperationIr::Cross(_)
            | FloatOperationIr::Quantize(_)
            | FloatOperationIr::Dequantize(_)
            | FloatOperationIr::GridSample2d(_) => None,
        }
    }

    fn int(&self, op: &IntOperationIr) -> Option<OpPlan> {
        match op {
            IntOperationIr::IntoFloat(desc) => {
                self.elementwise(Linearity::Nonlinear, [&desc.input], &desc.out)
            }
            IntOperationIr::BitwiseAnd(desc)
            | IntOperationIr::BitwiseOr(desc)
            | IntOperationIr::BitwiseXor(desc)
            | IntOperationIr::BitwiseLeftShift(desc)
            | IntOperationIr::BitwiseRightShift(desc) => self.binary(Linearity::Nonlinear, desc),
            IntOperationIr::BitwiseAndScalar(desc)
            | IntOperationIr::BitwiseOrScalar(desc)
            | IntOperationIr::BitwiseXorScalar(desc)
            | IntOperationIr::BitwiseLeftShiftScalar(desc)
            | IntOperationIr::BitwiseRightShiftScalar(desc) => {
                self.scalar(Linearity::Nonlinear, desc)
            }
            IntOperationIr::BitwiseNot(desc) => self.unary(Linearity::Nonlinear, desc),
            IntOperationIr::Matmul(desc) => self.matmul(desc),
        }
    }

    fn bool(&self, op: &BoolOperationIr) -> Option<OpPlan> {
        match op {
            BoolOperationIr::IntoFloat(desc) | BoolOperationIr::IntoInt(desc) => {
                self.elementwise(Linearity::Nonlinear, [&desc.input], &desc.out)
            }
            BoolOperationIr::Not(desc) => self.unary(Linearity::Nonlinear, desc),
            BoolOperationIr::And(desc) | BoolOperationIr::Or(desc) | BoolOperationIr::Xor(desc) => {
                self.binary(Linearity::Nonlinear, desc)
            }
        }
    }

    fn module(&self, op: &ModuleOperationIr) -> Option<OpPlan> {
        match op {
            ModuleOperationIr::Linear(desc) => self.linear(desc),
            ModuleOperationIr::LinearXBackward(desc) => self.linear_x_backward(desc),
            ModuleOperationIr::LinearWeightBackward(desc) => self.linear_weight_backward(desc),
            ModuleOperationIr::LinearBiasBackward(desc) => self.linear_bias_backward(desc),
            ModuleOperationIr::Embedding(desc) => self.embedding(desc),
            ModuleOperationIr::EmbeddingBackward(desc) => self.embedding_backward(desc),
            // No rule yet: these gather their inputs and run replicated.
            ModuleOperationIr::BatchNorm(_)
            | ModuleOperationIr::Conv1d(_)
            | ModuleOperationIr::Conv1dXBackward(_)
            | ModuleOperationIr::Conv1dWeightBackward(_)
            | ModuleOperationIr::Conv1dBiasBackward(_)
            | ModuleOperationIr::Conv2d(_)
            | ModuleOperationIr::Conv2dXBackward(_)
            | ModuleOperationIr::Conv2dWeightBackward(_)
            | ModuleOperationIr::Conv2dBiasBackward(_)
            | ModuleOperationIr::Conv3d(_)
            | ModuleOperationIr::Conv3dXBackward(_)
            | ModuleOperationIr::Conv3dWeightBackward(_)
            | ModuleOperationIr::Conv3dBiasBackward(_)
            | ModuleOperationIr::DeformableConv2d(_)
            | ModuleOperationIr::DeformableConv2dBackward(_)
            | ModuleOperationIr::ConvTranspose1d(_)
            | ModuleOperationIr::ConvTranspose2d(_)
            | ModuleOperationIr::ConvTranspose3d(_)
            | ModuleOperationIr::ConvTranspose1dWeightBackward(_)
            | ModuleOperationIr::ConvTranspose1dBiasBackward(_)
            | ModuleOperationIr::ConvTranspose2dWeightBackward(_)
            | ModuleOperationIr::ConvTranspose2dBiasBackward(_)
            | ModuleOperationIr::ConvTranspose3dWeightBackward(_)
            | ModuleOperationIr::ConvTranspose3dBiasBackward(_)
            | ModuleOperationIr::AvgPool1d(_)
            | ModuleOperationIr::AvgPool2d(_)
            | ModuleOperationIr::AvgPool3d(_)
            | ModuleOperationIr::AvgPool1dBackward(_)
            | ModuleOperationIr::AvgPool2dBackward(_)
            | ModuleOperationIr::AvgPool3dBackward(_)
            | ModuleOperationIr::AdaptiveAvgPool1d(_)
            | ModuleOperationIr::AdaptiveAvgPool2d(_)
            | ModuleOperationIr::AdaptiveAvgPool3d(_)
            | ModuleOperationIr::AdaptiveAvgPool1dBackward(_)
            | ModuleOperationIr::AdaptiveAvgPool2dBackward(_)
            | ModuleOperationIr::AdaptiveAvgPool3dBackward(_)
            | ModuleOperationIr::MaxPool1d(_)
            | ModuleOperationIr::MaxPool1dWithIndices(_)
            | ModuleOperationIr::MaxPool1dWithIndicesBackward(_)
            | ModuleOperationIr::MaxPool2d(_)
            | ModuleOperationIr::MaxPool2dWithIndices(_)
            | ModuleOperationIr::MaxPool2dWithIndicesBackward(_)
            | ModuleOperationIr::MaxPool3d(_)
            | ModuleOperationIr::MaxPool3dWithIndices(_)
            | ModuleOperationIr::MaxPool3dWithIndicesBackward(_)
            | ModuleOperationIr::Interpolate(_)
            | ModuleOperationIr::InterpolateBackward(_)
            | ModuleOperationIr::Attention(_)
            | ModuleOperationIr::CtcLoss(_)
            | ModuleOperationIr::CtcLossBackward(_)
            | ModuleOperationIr::LayerNorm(_)
            | ModuleOperationIr::Unfold4d(_) => None,
        }
    }

    fn activation(&self, op: &ActivationOperationIr) -> Option<OpPlan> {
        match op {
            ActivationOperationIr::Relu(desc)
            | ActivationOperationIr::Gelu(desc)
            | ActivationOperationIr::Sigmoid(desc)
            | ActivationOperationIr::LogSigmoid(desc) => self.unary(Linearity::Nonlinear, desc),
            ActivationOperationIr::LeakyRelu(desc) => self.scalar(Linearity::Nonlinear, desc),
            ActivationOperationIr::ReluBackward(desc)
            | ActivationOperationIr::GeluBackward(desc)
            | ActivationOperationIr::SigmoidBackward(desc)
            | ActivationOperationIr::LogSigmoidBackward(desc) => {
                self.binary(Linearity::LinearIn { input: GRADIENT }, desc)
            }
            ActivationOperationIr::Softmax(desc)
            | ActivationOperationIr::LogSoftmax(desc)
            | ActivationOperationIr::Softmin(desc) => {
                self.whole_dims(&desc.input, &desc.out, Reduction::Dim(desc.axis))
            }
            ActivationOperationIr::PRelu(_) | ActivationOperationIr::HardSigmoid(_) => None,
        }
    }

    fn unary(&self, linearity: Linearity, desc: &UnaryOpIr) -> Option<OpPlan> {
        self.elementwise(linearity, [&desc.input], &desc.out)
    }

    fn scalar(&self, linearity: Linearity, desc: &ScalarOpIr) -> Option<OpPlan> {
        self.elementwise(linearity, [&desc.lhs], &desc.out)
    }

    fn binary(&self, linearity: Linearity, desc: &BinaryOpIr) -> Option<OpPlan> {
        self.elementwise(linearity, [&desc.lhs, &desc.rhs], &desc.out)
    }

    /// Broadcasting lines up dims of the same index, so an input of another member has no rule.
    fn elementwise<const N: usize>(
        &self,
        linearity: Linearity,
        inputs: [&TensorIr; N],
        out: &TensorIr,
    ) -> Option<OpPlan> {
        let num_dims = out.shape.num_dims();
        if inputs
            .iter()
            .any(|input| input.shape.num_dims() != num_dims)
        {
            return None;
        }
        let placement = linearity.placement(
            inputs.map(|input| self.placement(input)),
            inputs.map(|input| &input.shape),
        );
        Targets::of(inputs, placement.inputs)
            .output(out, placement.output)
            .every_member()
    }

    fn matmul(&self, desc: &MatmulOpIr) -> Option<OpPlan> {
        let placement = MatmulRule::new(
            self.placement(&desc.lhs),
            self.placement(&desc.rhs),
            &desc.lhs.shape,
            &desc.rhs.shape,
        )
        .placement();
        self.ruled([&desc.lhs, &desc.rhs], &desc.out, placement)
    }

    fn linear(&self, desc: &LinearOpIr) -> Option<OpPlan> {
        let rule = LinearRule::new(
            self.placement(&desc.x),
            self.placement(&desc.weight),
            &desc.x.shape,
            &desc.weight.shape,
        );
        let OpPlacement {
            inputs: [x, weight, bias],
            output,
        } = rule.placement();
        let mut targets =
            Targets::of([&desc.x, &desc.weight], [x, weight]).output(&desc.out, output);
        let Some(bias_tensor) = &desc.bias else {
            return targets.every_member();
        };
        targets = targets.input(bias_tensor, bias);
        match rule.bias_on_one_member() {
            true => targets.build(Execution::BiasOnFirstMember),
            false => targets.every_member(),
        }
    }

    fn linear_x_backward(&self, desc: &LinearXBackwardOpIr) -> Option<OpPlan> {
        let placement = LinearRule::x_backward(
            self.placement(&desc.weight),
            self.placement(&desc.output_grad),
            &desc.weight.shape,
            &desc.output_grad.shape,
        );
        self.ruled([&desc.weight, &desc.output_grad], &desc.out, placement)
    }

    fn linear_weight_backward(&self, desc: &LinearWeightBackwardOpIr) -> Option<OpPlan> {
        let placement = LinearRule::weight_backward(
            self.placement(&desc.x),
            self.placement(&desc.output_grad),
            &desc.x.shape,
            &desc.output_grad.shape,
        );
        self.ruled([&desc.x, &desc.output_grad], &desc.out, placement)
    }

    fn linear_bias_backward(&self, desc: &LinearBiasBackwardOpIr) -> Option<OpPlan> {
        let placement =
            LinearRule::bias_backward(self.placement(&desc.output_grad), &desc.output_grad.shape);
        self.ruled([&desc.output_grad], &desc.out, placement)
    }

    fn embedding(&self, desc: &EmbeddingOpIr) -> Option<OpPlan> {
        let rule = EmbeddingRule::new(self.placement(&desc.weights), self.placement(&desc.indices));
        let placement = rule.placement();
        let targets = Targets::of([&desc.weights, &desc.indices], placement.inputs)
            .output(&desc.out, placement.output);
        match rule {
            EmbeddingRule::Vocab => targets.build(Execution::VocabLookup {
                split: self.vocab_split(&desc.weights),
            }),
            _ => targets.every_member(),
        }
    }

    fn embedding_backward(&self, desc: &EmbeddingBackwardOpIr) -> Option<OpPlan> {
        let rule = EmbeddingBackwardRule::new(
            self.placement(&desc.weights),
            self.placement(&desc.out_grad),
            self.placement(&desc.indices),
        );
        let placement = rule.placement();
        let targets = Targets::of(
            [&desc.weights, &desc.out_grad, &desc.indices],
            placement.inputs,
        )
        .output(&desc.out, placement.output);
        match rule {
            EmbeddingBackwardRule::Vocab => targets.build(Execution::VocabBackward {
                split: self.vocab_split(&desc.weights),
            }),
            _ => targets.every_member(),
        }
    }

    fn vocab_split(&self, weights: &TensorIr) -> DimSplit {
        DimSplit::new(weights.shape[EmbeddingRule::VOCAB_DIM], self.members)
    }

    fn sum(&self, input: &TensorIr, out: &TensorIr, reduction: Reduction) -> Option<OpPlan> {
        let placement = ReduceRule::new(self.placement(input), reduction).placement();
        self.ruled([input], out, placement)
    }

    /// An integer mean rounds each summand, so only a float mean reduces across members.
    fn mean(
        &self,
        dtype: DType,
        input: &TensorIr,
        out: &TensorIr,
        reduction: Reduction,
    ) -> Option<OpPlan> {
        if !dtype.is_float() {
            return self.whole_dims(input, out, reduction);
        }
        let rule = ReduceRule::new(self.placement(input), reduction);
        let placement = rule.placement();
        let targets = Targets::of([input], placement.inputs).output(out, placement.output);
        match rule {
            ReduceRule::Local { .. } => targets.every_member(),
            ReduceRule::AcrossMembers { dim } => {
                let count = match reduction {
                    Reduction::All => input.shape.num_elements(),
                    Reduction::Dim(_) => input.shape[dim],
                };
                targets.build(Execution::SumThenDivide { count })
            }
        }
    }

    fn whole_dims(&self, input: &TensorIr, out: &TensorIr, reads: Reduction) -> Option<OpPlan> {
        let placement = WholeDimRule::new(self.placement(input), reads).placement();
        self.ruled([input], out, placement)
    }

    fn reshape(&self, desc: &ShapeOpIr) -> Option<OpPlan> {
        let placement = ReshapeRule::new(
            self.placement(&desc.input),
            &desc.input.shape,
            &desc.out.shape,
            self.members,
        )
        .placement();
        self.ruled([&desc.input], &desc.out, placement)
    }

    fn permute(&self, input: &TensorIr, out: &TensorIr, axes: &[usize]) -> Option<OpPlan> {
        let placement = self.placement(input);
        Targets::of([input], [placement])
            .output(out, placement.permuted(axes))
            .every_member()
    }

    fn flip(&self, desc: &FlipOpIr) -> Option<OpPlan> {
        let placement = self.placement(&desc.input);
        if let GroupPlacement::Sharded { dim } = placement
            && desc.axes.contains(&dim)
        {
            return None;
        }
        self.kept(&desc.input, &desc.out)
    }

    /// Only new leading dims or dims of length 1 are expanded, and neither can be split.
    fn expand(&self, desc: &ShapeOpIr) -> Option<OpPlan> {
        let placement = self.placement(&desc.input);
        let output =
            placement.with_num_dims(desc.input.shape.num_dims(), desc.out.shape.num_dims());
        Targets::of([&desc.input], [placement])
            .output(&desc.out, output)
            .every_member()
    }

    /// Slicing is linear, and a split dim taken whole is each member's whole chunk.
    fn slice(&self, desc: &SliceOpIr) -> Option<OpPlan> {
        let placement = self.placement(&desc.tensor);
        if let GroupPlacement::Sharded { dim } = placement {
            let whole = desc.ranges.get(dim).is_none_or(|range| {
                range.step == 1 && desc.out.shape[dim] == desc.tensor.shape[dim]
            });
            if !whole {
                return None;
            }
        }
        self.kept(&desc.tensor, &desc.out)
    }

    fn cat(&self, desc: &CatOpIr) -> Option<OpPlan> {
        let placements: Vec<GroupPlacement> =
            desc.tensors.iter().map(|t| self.placement(t)).collect();
        let target = if placements
            .iter()
            .all(|placement| *placement == GroupPlacement::Partial)
        {
            GroupPlacement::Partial
        } else {
            placements.iter().copied().find(|placement| {
                matches!(placement, GroupPlacement::Sharded { dim } if *dim != desc.dim)
            })?
        };
        desc.tensors
            .iter()
            .fold(Targets::default(), |targets, tensor| {
                targets.input(tensor, target)
            })
            .output(&desc.out, target)
            .every_member()
    }

    fn created(&self, out: &TensorIr) -> Option<OpPlan> {
        Targets::default()
            .output(out, GroupPlacement::Replicated)
            .every_member()
    }

    fn random(&self, out: &TensorIr) -> Option<OpPlan> {
        Targets::default()
            .output(out, GroupPlacement::Replicated)
            .build(Execution::FirstMemberCopied)
    }

    fn kept(&self, input: &TensorIr, out: &TensorIr) -> Option<OpPlan> {
        let placement = self.placement(input);
        Targets::of([input], [placement])
            .output(out, placement)
            .every_member()
    }

    fn ruled<const N: usize>(
        &self,
        inputs: [&TensorIr; N],
        out: &TensorIr,
        placement: OpPlacement<N>,
    ) -> Option<OpPlan> {
        Targets::of(inputs, placement.inputs)
            .output(out, placement.output)
            .every_member()
    }

    fn placement(&self, tensor: &TensorIr) -> GroupPlacement {
        *self
            .placements
            .get(&tensor.id)
            .unwrap_or_else(|| panic!("{} has no placement on its device group", tensor.id))
    }
}

/// The placements an op's tensors take, refusing an input asked to be in two places at once.
#[derive(Default)]
struct Targets {
    inputs: HashMap<TensorId, GroupPlacement>,
    outputs: HashMap<TensorId, GroupPlacement>,
    conflict: bool,
}

impl Targets {
    fn of<const N: usize>(tensors: [&TensorIr; N], placements: [GroupPlacement; N]) -> Self {
        tensors
            .into_iter()
            .zip(placements)
            .fold(Self::default(), |targets, (tensor, placement)| {
                targets.input(tensor, placement)
            })
    }

    fn input(mut self, tensor: &TensorIr, placement: GroupPlacement) -> Self {
        let previous = self.inputs.insert(tensor.id, placement);
        self.conflict |= previous.is_some_and(|previous| previous != placement);
        self
    }

    fn output(mut self, tensor: &TensorIr, placement: GroupPlacement) -> Self {
        self.outputs.insert(tensor.id, placement);
        self
    }

    fn every_member(self) -> Option<OpPlan> {
        self.build(Execution::EveryMember)
    }

    fn build(self, execution: Execution) -> Option<OpPlan> {
        (!self.conflict).then_some(OpPlan {
            inputs: self.inputs,
            outputs: self.outputs,
            execution,
        })
    }
}

/// The input of an activation's backward that holds the incoming gradient.
const GRADIENT: usize = 1;
