use std::collections::HashMap;

use burn_backend::{DType, ops::attention::AttentionShapes};
use burn_ir::{
    ActivationOperationIr, AttentionOpIr, BaseOperationIr, BinaryOpIr, FloatOperationIr,
    MatmulOpIr, ModuleOperationIr, NumericOperationIr, OperationIr, ScalarOpIr, ShapeOpIr,
    SliceAssignOpIr, TensorId, TensorIr, UnaryOpIr,
};

use crate::{
    AttentionRule, ElementwiseRule, LayoutRule, MatmulRule, OpSharding, ReducedDims, ReshapeRule,
    Sharding, SumRule,
};

/// Where each tensor of one op must be for the op to run on every member, where its outputs land,
/// and when the members run it.
#[derive(Debug)]
pub struct OpPlan {
    input_shardings: HashMap<TensorId, Sharding>,
    output_shardings: HashMap<TensorId, Sharding>,
    execution: Execution,
}

/// When the members run one op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Execution {
    /// Now, each member on its own shards.
    Eager,
    /// When its output is first needed: split if that use splits it, else on member 0 and copied.
    Deferred,
}

impl OpPlan {
    /// An op without a rule, or whose inputs would need two shardings at once, gathers every
    /// input and runs whole on every member.
    pub fn new(
        op: &OperationIr,
        shardings: &HashMap<TensorId, Sharding>,
        num_members: usize,
    ) -> Self {
        TensorShardings {
            shardings,
            num_members,
        }
        .plan(op)
        .unwrap_or_else(|| Self::gathered(op))
    }

    pub fn input_sharding(&self, id: &TensorId) -> Sharding {
        self.input_shardings[id]
    }

    pub fn output_shardings(&self) -> &HashMap<TensorId, Sharding> {
        &self.output_shardings
    }

    pub fn execution(&self) -> Execution {
        self.execution
    }

    /// The plan a rule's `sharding` asks for, or `None` when one tensor, read as two inputs, is
    /// asked for two shardings, or when a quantized tensor would not be replicated: its shards
    /// cannot be sliced, concatenated or summed.
    fn from_rule<const N: usize>(
        inputs: [&TensorIr; N],
        out: &TensorIr,
        sharding: OpSharding<N>,
        execution: Execution,
    ) -> Option<Self> {
        if inputs
            .into_iter()
            .zip(sharding.inputs)
            .chain([(out, sharding.output)])
            .any(|(tensor, target)| {
                matches!(tensor.dtype, DType::QFloat(_)) && target != Sharding::Replicated
            })
        {
            return None;
        }
        let mut input_shardings = HashMap::new();
        for (tensor, target) in inputs.into_iter().zip(sharding.inputs) {
            if input_shardings
                .insert(tensor.id, target)
                .is_some_and(|previous| previous != target)
            {
                return None;
            }
        }
        Some(Self {
            input_shardings,
            output_shardings: HashMap::from([(out.id, sharding.output)]),
            execution,
        })
    }

    fn gathered(op: &OperationIr) -> Self {
        let replicated = |tensor: &TensorIr| (tensor.id, Sharding::Replicated);
        Self {
            input_shardings: op.inputs().map(replicated).collect(),
            output_shardings: op.outputs().map(replicated).collect(),
            execution: Execution::Eager,
        }
    }
}

/// The shardings of a sharded compute device's tensors, which pick the rule each op runs by.
struct TensorShardings<'a> {
    shardings: &'a HashMap<TensorId, Sharding>,
    num_members: usize,
}

impl TensorShardings<'_> {
    fn plan(&self, op: &OperationIr) -> Option<OpPlan> {
        match op {
            OperationIr::BaseFloat(BaseOperationIr::Reshape(desc))
            | OperationIr::BaseInt(BaseOperationIr::Reshape(desc))
            | OperationIr::BaseBool(BaseOperationIr::Reshape(desc)) => self.reshape(desc),
            OperationIr::BaseFloat(BaseOperationIr::SwapDims(desc))
            | OperationIr::BaseInt(BaseOperationIr::SwapDims(desc))
            | OperationIr::BaseBool(BaseOperationIr::SwapDims(desc)) => self.layout(
                &desc.input,
                &desc.out,
                LayoutRule::swap(self.sharding(&desc.input), desc.dim1, desc.dim2),
            ),
            OperationIr::BaseFloat(BaseOperationIr::Permute(desc))
            | OperationIr::BaseInt(BaseOperationIr::Permute(desc))
            | OperationIr::BaseBool(BaseOperationIr::Permute(desc)) => self.layout(
                &desc.input,
                &desc.out,
                LayoutRule::permute(self.sharding(&desc.input), &desc.axes),
            ),
            OperationIr::BaseFloat(BaseOperationIr::Slice(desc))
            | OperationIr::BaseInt(BaseOperationIr::Slice(desc))
            | OperationIr::BaseBool(BaseOperationIr::Slice(desc)) => self.layout(
                &desc.tensor,
                &desc.out,
                LayoutRule::slice(
                    self.sharding(&desc.tensor),
                    &desc.ranges,
                    &desc.tensor.shape,
                ),
            ),
            OperationIr::BaseFloat(BaseOperationIr::SliceAssign(desc))
            | OperationIr::BaseInt(BaseOperationIr::SliceAssign(desc))
            | OperationIr::BaseBool(BaseOperationIr::SliceAssign(desc)) => self.slice_assign(desc),
            OperationIr::BaseFloat(BaseOperationIr::Expand(desc))
            | OperationIr::BaseInt(BaseOperationIr::Expand(desc))
            | OperationIr::BaseBool(BaseOperationIr::Expand(desc)) => self.layout(
                &desc.input,
                &desc.out,
                LayoutRule::expand(
                    self.sharding(&desc.input),
                    desc.input.shape.num_dims(),
                    desc.out.shape.num_dims(),
                ),
            ),
            OperationIr::NumericFloat(_, op) => self.numeric(op),
            OperationIr::Float(_, FloatOperationIr::Matmul(desc)) => self.matmul(desc),
            OperationIr::Float(_, FloatOperationIr::Random(desc)) => self.random(&desc.out),
            OperationIr::NumericInt(_, NumericOperationIr::IntRandom(desc)) => {
                self.random(&desc.out)
            }
            OperationIr::Activation(
                ActivationOperationIr::Relu(desc)
                | ActivationOperationIr::Gelu(desc)
                | ActivationOperationIr::Sigmoid(desc),
            ) => self.unary(ElementwiseRule::Nonlinear, desc),
            OperationIr::Module(ModuleOperationIr::Attention(desc)) => self.attention(desc),
            _ => None,
        }
    }

    fn numeric(&self, op: &NumericOperationIr) -> Option<OpPlan> {
        match op {
            NumericOperationIr::Add(desc) | NumericOperationIr::Sub(desc) => {
                self.binary(ElementwiseRule::Linear, desc)
            }
            NumericOperationIr::Mul(desc) => self.binary(ElementwiseRule::Multilinear, desc),
            NumericOperationIr::Div(desc) => {
                self.binary(ElementwiseRule::LinearIn { input: 0 }, desc)
            }
            NumericOperationIr::MulScalar(desc) | NumericOperationIr::DivScalar(desc) => {
                self.scalar(ElementwiseRule::Linear, desc)
            }
            NumericOperationIr::Neg(desc) => self.unary(ElementwiseRule::Linear, desc),
            NumericOperationIr::Sum(desc) => self.sum(&desc.input, &desc.out, ReducedDims::All),
            NumericOperationIr::SumDim(desc) => {
                self.sum(&desc.input, &desc.out, ReducedDims::Dim(desc.axis))
            }
            _ => None,
        }
    }

    fn unary(&self, rule: ElementwiseRule, desc: &UnaryOpIr) -> Option<OpPlan> {
        self.elementwise(rule, [&desc.input], &desc.out)
    }

    fn scalar(&self, rule: ElementwiseRule, desc: &ScalarOpIr) -> Option<OpPlan> {
        self.elementwise(rule, [&desc.lhs], &desc.out)
    }

    fn binary(&self, rule: ElementwiseRule, desc: &BinaryOpIr) -> Option<OpPlan> {
        self.elementwise(rule, [&desc.lhs, &desc.rhs], &desc.out)
    }

    /// Broadcasting lines up dims of the same index, so an input with another number of dims has
    /// no rule.
    fn elementwise<const N: usize>(
        &self,
        rule: ElementwiseRule,
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
        let sharding = rule.sharding(
            inputs.map(|input| self.sharding(input)),
            inputs.map(|input| &input.shape),
        );
        OpPlan::from_rule(inputs, out, sharding, Execution::Eager)
    }

    fn matmul(&self, desc: &MatmulOpIr) -> Option<OpPlan> {
        let sharding = MatmulRule::new(
            self.sharding(&desc.lhs),
            self.sharding(&desc.rhs),
            desc.lhs.shape.num_dims(),
        )
        .sharding();
        OpPlan::from_rule(
            [&desc.lhs, &desc.rhs],
            &desc.out,
            sharding,
            Execution::Eager,
        )
    }

    fn attention(&self, desc: &AttentionOpIr) -> Option<OpPlan> {
        let (query, key, value) = (&desc.query, &desc.key, &desc.value);
        let rule = AttentionRule::new(
            [query, key, value].map(|input| self.sharding(input)),
            &AttentionShapes::new(&query.shape, &key.shape, &value.shape),
            self.num_members,
        );
        let out = &desc.out;
        match (&desc.mask, &desc.attn_bias) {
            (None, None) => self.attend(rule, [query, key, value], out),
            (Some(scores), None) | (None, Some(scores)) => {
                self.attend(rule, [query, key, value, scores], out)
            }
            (Some(mask), Some(bias)) => self.attend(rule, [query, key, value, mask, bias], out),
        }
    }

    fn attend<const N: usize>(
        &self,
        rule: AttentionRule,
        inputs: [&TensorIr; N],
        out: &TensorIr,
    ) -> Option<OpPlan> {
        let sharding = rule.sharding(inputs.map(|input| &input.shape));
        OpPlan::from_rule(inputs, out, sharding, Execution::Eager)
    }

    fn sum(&self, input: &TensorIr, out: &TensorIr, reduced_dims: ReducedDims) -> Option<OpPlan> {
        let sharding = SumRule::new(self.sharding(input), reduced_dims).sharding();
        OpPlan::from_rule([input], out, sharding, Execution::Eager)
    }

    fn reshape(&self, desc: &ShapeOpIr) -> Option<OpPlan> {
        let sharding = ReshapeRule::new(
            self.sharding(&desc.input),
            &desc.input.shape,
            &desc.out.shape,
            self.num_members,
        )
        .sharding();
        OpPlan::from_rule([&desc.input], &desc.out, sharding, Execution::Eager)
    }

    fn layout(&self, input: &TensorIr, out: &TensorIr, rule: LayoutRule) -> Option<OpPlan> {
        OpPlan::from_rule([input], out, rule.sharding(), Execution::Eager)
    }

    fn slice_assign(&self, desc: &SliceAssignOpIr) -> Option<OpPlan> {
        let rule = LayoutRule::slice_assign(
            self.sharding(&desc.tensor),
            self.sharding(&desc.value),
            &desc.ranges,
            &desc.tensor.shape,
        );
        OpPlan::from_rule(
            [&desc.tensor, &desc.value],
            &desc.out,
            rule.sharding(),
            Execution::Eager,
        )
    }

    fn random(&self, out: &TensorIr) -> Option<OpPlan> {
        let sharding = OpSharding {
            inputs: [],
            output: Sharding::Replicated,
        };
        OpPlan::from_rule([], out, sharding, Execution::Deferred)
    }

    fn sharding(&self, tensor: &TensorIr) -> Sharding {
        self.shardings[&tensor.id]
    }
}
