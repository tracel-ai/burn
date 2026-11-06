use std::slice;

use burn_backend::DType;
use burn_ir::{
    ActivationOperationIr, AttentionOpIr, BaseOperationIr, BinaryOpIr, BoolOperationIr, CastOpIr,
    CustomOpIr, FloatOperationIr, IntOperationIr, MatmulOpIr, ModuleOperationIr,
    NumericOperationIr, OperationIr, ReduceDimOpIr, ReduceOpIr, ScalarOpIr, TensorIr, UnaryOpIr,
};

use crate::{Linearity, OpInputs, OpKind, ReducedDims, ReductionKind};

/// How the planner reads an op: by its kind and the tensors it reads, or not at all.
#[derive(Debug, PartialEq)]
pub enum OpClass<'a> {
    Kind {
        kind: OpKind<'a>,
        inputs: OpInputs<&'a TensorIr>,
        out: &'a TensorIr,
    },
    /// A custom op, which its rule classifies.
    Custom(&'a CustomOpIr),
    /// No rule: the op gathers its inputs and runs whole on every member.
    Gathered,
}

/// Every op family but the modules is matched without a catch-all, so a new op cannot compile
/// until it is given a class.
impl<'a> From<&'a OperationIr> for OpClass<'a> {
    fn from(op: &'a OperationIr) -> Self {
        match op {
            OperationIr::BaseFloat(op) | OperationIr::BaseInt(op) | OperationIr::BaseBool(op) => {
                Self::base(op)
            }
            OperationIr::NumericFloat(dtype, op) | OperationIr::NumericInt(dtype, op) => {
                Self::numeric(*dtype, op)
            }
            OperationIr::Float(_, op) => Self::float(op),
            OperationIr::Int(op) => Self::int(op),
            OperationIr::Bool(op) => Self::bool(op),
            OperationIr::Activation(op) => Self::activation(op),
            OperationIr::Module(op) => Self::module(op),
            OperationIr::Custom(op) => Self::Custom(op),
            OperationIr::Init(_) | OperationIr::Drop(_) => Self::Gathered,
        }
    }
}

impl<'a> OpClass<'a> {
    fn base(op: &'a BaseOperationIr) -> Self {
        match op {
            BaseOperationIr::Reshape(desc) => {
                Self::of_kind(OpKind::Reshape, [&desc.input], &desc.out)
            }
            BaseOperationIr::SwapDims(desc) => Self::of_kind(
                OpKind::SwapDims {
                    dim1: desc.dim1,
                    dim2: desc.dim2,
                },
                [&desc.input],
                &desc.out,
            ),
            BaseOperationIr::Permute(desc) => Self::of_kind(
                OpKind::Permute { axes: &desc.axes },
                [&desc.input],
                &desc.out,
            ),
            BaseOperationIr::Cat(desc) => Self::Kind {
                kind: OpKind::Concat { dim: desc.dim },
                inputs: desc.tensors.iter().collect(),
                out: &desc.out,
            },
            BaseOperationIr::Slice(desc) => Self::of_kind(
                OpKind::Slice {
                    ranges: &desc.ranges,
                },
                [&desc.tensor],
                &desc.out,
            ),
            BaseOperationIr::SliceAssign(desc) => Self::of_kind(
                OpKind::SliceAssign {
                    ranges: &desc.ranges,
                },
                [&desc.tensor, &desc.value],
                &desc.out,
            ),
            BaseOperationIr::Expand(desc) => {
                Self::of_kind(OpKind::Expand, [&desc.input], &desc.out)
            }
            BaseOperationIr::Equal(desc) | BaseOperationIr::NotEqual(desc) => {
                Self::binary(Linearity::Nonlinear, desc)
            }
            BaseOperationIr::EqualElem(desc) | BaseOperationIr::NotEqualElem(desc) => {
                Self::scalar(Linearity::Nonlinear, desc)
            }
            BaseOperationIr::Cast(desc) => Self::cast(desc),
            BaseOperationIr::MaskWhere(desc) => Self::of_kind(
                OpKind::Elementwise(Linearity::Nonlinear),
                [&desc.tensor, &desc.mask, &desc.value],
                &desc.out,
            ),
            BaseOperationIr::MaskFill(desc) => Self::of_kind(
                OpKind::Elementwise(Linearity::Nonlinear),
                [&desc.tensor, &desc.mask],
                &desc.out,
            ),
            BaseOperationIr::All(desc) | BaseOperationIr::Any(desc) => {
                Self::reduce_all(desc, ReductionKind::Nonlinear)
            }
            BaseOperationIr::AllDim(desc) | BaseOperationIr::AnyDim(desc) => {
                Self::reduce_dim(desc, ReductionKind::Nonlinear)
            }
            BaseOperationIr::Flip(_)
            | BaseOperationIr::Unfold(_)
            | BaseOperationIr::Select(_)
            | BaseOperationIr::SelectAssign(_)
            | BaseOperationIr::Gather(_)
            | BaseOperationIr::Scatter(_)
            | BaseOperationIr::ScatterNd(_)
            | BaseOperationIr::GatherNd(_)
            | BaseOperationIr::RepeatDim(_)
            | BaseOperationIr::Empty(_)
            | BaseOperationIr::Ones(_)
            | BaseOperationIr::Zeros(_) => Self::Gathered,
        }
    }

    /// Integer division and mean truncate, so neither distributes over a sum.
    fn numeric(dtype: DType, op: &'a NumericOperationIr) -> Self {
        let (division, scalar_division, mean) = match dtype.is_float() {
            true => (
                Linearity::LinearIn { input: 0 },
                Linearity::Linear,
                ReductionKind::Mean,
            ),
            false => (
                Linearity::Nonlinear,
                Linearity::Nonlinear,
                ReductionKind::Nonlinear,
            ),
        };
        match op {
            NumericOperationIr::Add(desc) | NumericOperationIr::Sub(desc) => {
                Self::binary(Linearity::Linear, desc)
            }
            NumericOperationIr::Mul(desc) => Self::binary(Linearity::Multilinear, desc),
            NumericOperationIr::Div(desc) => Self::binary(division, desc),
            NumericOperationIr::Rem(desc)
            | NumericOperationIr::Powi(desc)
            | NumericOperationIr::Greater(desc)
            | NumericOperationIr::GreaterEqual(desc)
            | NumericOperationIr::Lower(desc)
            | NumericOperationIr::LowerEqual(desc) => Self::binary(Linearity::Nonlinear, desc),
            NumericOperationIr::MulScalar(desc) => Self::scalar(Linearity::Linear, desc),
            NumericOperationIr::DivScalar(desc) => Self::scalar(scalar_division, desc),
            NumericOperationIr::AddScalar(desc)
            | NumericOperationIr::SubScalar(desc)
            | NumericOperationIr::RemScalar(desc)
            | NumericOperationIr::PowiScalar(desc)
            | NumericOperationIr::GreaterElem(desc)
            | NumericOperationIr::GreaterEqualElem(desc)
            | NumericOperationIr::LowerElem(desc)
            | NumericOperationIr::LowerEqualElem(desc)
            | NumericOperationIr::ClampMin(desc)
            | NumericOperationIr::ClampMax(desc) => Self::scalar(Linearity::Nonlinear, desc),
            NumericOperationIr::Neg(desc) => Self::unary(Linearity::Linear, desc),
            NumericOperationIr::Abs(desc) | NumericOperationIr::Sign(desc) => {
                Self::unary(Linearity::Nonlinear, desc)
            }
            NumericOperationIr::Clamp(desc) => Self::of_kind(
                OpKind::Elementwise(Linearity::Nonlinear),
                [&desc.tensor],
                &desc.out,
            ),
            NumericOperationIr::Sum(desc) => Self::reduce_all(desc, ReductionKind::Sum),
            NumericOperationIr::Mean(desc) => Self::reduce_all(desc, mean),
            NumericOperationIr::Prod(desc)
            | NumericOperationIr::Max(desc)
            | NumericOperationIr::Min(desc)
            | NumericOperationIr::MaxAbs(desc) => Self::reduce_all(desc, ReductionKind::Nonlinear),
            NumericOperationIr::SumDim(desc) => Self::reduce_dim(desc, ReductionKind::Sum),
            NumericOperationIr::MeanDim(desc) => Self::reduce_dim(desc, mean),
            NumericOperationIr::ProdDim(desc)
            | NumericOperationIr::MaxDim(desc)
            | NumericOperationIr::MinDim(desc)
            | NumericOperationIr::MaxAbsDim(desc)
            | NumericOperationIr::ArgMax(desc)
            | NumericOperationIr::ArgMin(desc) => Self::reduce_dim(desc, ReductionKind::Nonlinear),
            NumericOperationIr::SumDims(desc) => {
                let dims = ReducedDims::Dims(&desc.axes);
                Self::of_kind(
                    OpKind::Reduction {
                        dims,
                        kind: ReductionKind::Sum,
                    },
                    [&desc.input],
                    &desc.out,
                )
            }
            NumericOperationIr::TopK(desc) | NumericOperationIr::ArgTopK(desc) => Self::of_kind(
                OpKind::AlongDim { dim: desc.axis },
                [&desc.input],
                &desc.out,
            ),
            NumericOperationIr::CumSum(desc)
            | NumericOperationIr::CumProd(desc)
            | NumericOperationIr::CumMin(desc)
            | NumericOperationIr::CumMax(desc) => Self::of_kind(
                OpKind::AlongDim { dim: desc.axis },
                [&desc.input],
                &desc.out,
            ),
            NumericOperationIr::Sort(desc) | NumericOperationIr::ArgSort(desc) => {
                Self::of_kind(OpKind::AlongDim { dim: desc.dim }, [&desc.input], &desc.out)
            }
            NumericOperationIr::IntRandom(desc) => Self::of_kind(OpKind::Random, [], &desc.out),
            NumericOperationIr::Full(_)
            | NumericOperationIr::TopKWithIndices(_)
            | NumericOperationIr::MaxDimWithIndices(_)
            | NumericOperationIr::MinDimWithIndices(_)
            | NumericOperationIr::SortWithIndices(_)
            | NumericOperationIr::Pad(_) => Self::Gathered,
        }
    }

    fn float(op: &'a FloatOperationIr) -> Self {
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
            | FloatOperationIr::IsInf(desc) => Self::unary(Linearity::Nonlinear, desc),
            FloatOperationIr::PowfScalar(desc) | FloatOperationIr::FmodScalar(desc) => {
                Self::scalar(Linearity::Nonlinear, desc)
            }
            FloatOperationIr::ArcTan2(desc)
            | FloatOperationIr::Powf(desc)
            | FloatOperationIr::Hypot(desc)
            | FloatOperationIr::Fmod(desc) => Self::binary(Linearity::Nonlinear, desc),
            FloatOperationIr::IntoInt(desc) => Self::cast(desc),
            FloatOperationIr::Matmul(desc) => Self::matmul(desc),
            FloatOperationIr::Random(desc) => Self::of_kind(OpKind::Random, [], &desc.out),
            // A member's shard holds whole quantization blocks, so it dequantizes to its own chunk.
            FloatOperationIr::Dequantize(desc) => Self::of_kind(
                OpKind::Elementwise(Linearity::Nonlinear),
                [&desc.input],
                &desc.out,
            ),
            FloatOperationIr::Cross(_)
            | FloatOperationIr::Quantize(_)
            | FloatOperationIr::GridSample2d(_) => Self::Gathered,
        }
    }

    fn int(op: &'a IntOperationIr) -> Self {
        match op {
            IntOperationIr::IntoFloat(desc) => Self::cast(desc),
            IntOperationIr::BitwiseAnd(desc)
            | IntOperationIr::BitwiseOr(desc)
            | IntOperationIr::BitwiseXor(desc)
            | IntOperationIr::BitwiseLeftShift(desc)
            | IntOperationIr::BitwiseRightShift(desc) => Self::binary(Linearity::Nonlinear, desc),
            IntOperationIr::BitwiseAndScalar(desc)
            | IntOperationIr::BitwiseOrScalar(desc)
            | IntOperationIr::BitwiseXorScalar(desc)
            | IntOperationIr::BitwiseLeftShiftScalar(desc)
            | IntOperationIr::BitwiseRightShiftScalar(desc) => {
                Self::scalar(Linearity::Nonlinear, desc)
            }
            IntOperationIr::BitwiseNot(desc) => Self::unary(Linearity::Nonlinear, desc),
            IntOperationIr::Matmul(desc) => Self::matmul(desc),
        }
    }

    fn bool(op: &'a BoolOperationIr) -> Self {
        match op {
            BoolOperationIr::IntoFloat(desc) | BoolOperationIr::IntoInt(desc) => Self::cast(desc),
            BoolOperationIr::Not(desc) => Self::unary(Linearity::Nonlinear, desc),
            BoolOperationIr::And(desc) | BoolOperationIr::Or(desc) | BoolOperationIr::Xor(desc) => {
                Self::binary(Linearity::Nonlinear, desc)
            }
        }
    }

    /// A backward is linear in the upstream gradient, its `rhs`, with the forward's tensor fixed.
    fn activation(op: &'a ActivationOperationIr) -> Self {
        match op {
            ActivationOperationIr::Relu(desc)
            | ActivationOperationIr::Gelu(desc)
            | ActivationOperationIr::Sigmoid(desc)
            | ActivationOperationIr::LogSigmoid(desc) => Self::unary(Linearity::Nonlinear, desc),
            ActivationOperationIr::LeakyRelu(desc) => Self::scalar(Linearity::Nonlinear, desc),
            ActivationOperationIr::PRelu(desc) => Self::binary(Linearity::Nonlinear, desc),
            ActivationOperationIr::HardSigmoid(desc) => Self::of_kind(
                OpKind::Elementwise(Linearity::Nonlinear),
                [&desc.tensor],
                &desc.out,
            ),
            ActivationOperationIr::ReluBackward(desc)
            | ActivationOperationIr::GeluBackward(desc)
            | ActivationOperationIr::SigmoidBackward(desc)
            | ActivationOperationIr::LogSigmoidBackward(desc) => {
                Self::binary(Linearity::LinearIn { input: 1 }, desc)
            }
            ActivationOperationIr::Softmax(desc)
            | ActivationOperationIr::LogSoftmax(desc)
            | ActivationOperationIr::Softmin(desc) => Self::of_kind(
                OpKind::AlongDim { dim: desc.axis },
                [&desc.input],
                &desc.out,
            ),
        }
    }

    /// A module op has no rule unless it is given one here: new ones gather.
    fn module(op: &'a ModuleOperationIr) -> Self {
        match op {
            // An attention over a key/value cache has no rule yet: its pools and page table are
            // not laid out by sequence, so it gathers, as it did as an op of its own.
            ModuleOperationIr::Attention(desc) if desc.lengths.is_some() => Self::Gathered,
            ModuleOperationIr::Attention(desc) => Self::attention(desc),
            _ => Self::Gathered,
        }
    }

    fn unary(linearity: Linearity, desc: &'a UnaryOpIr) -> Self {
        Self::of_kind(OpKind::Elementwise(linearity), [&desc.input], &desc.out)
    }

    fn scalar(linearity: Linearity, desc: &'a ScalarOpIr) -> Self {
        Self::of_kind(OpKind::Elementwise(linearity), [&desc.lhs], &desc.out)
    }

    fn binary(linearity: Linearity, desc: &'a BinaryOpIr) -> Self {
        Self::of_kind(
            OpKind::Elementwise(linearity),
            [&desc.lhs, &desc.rhs],
            &desc.out,
        )
    }

    fn cast(desc: &'a CastOpIr) -> Self {
        Self::of_kind(
            OpKind::Elementwise(Linearity::Nonlinear),
            [&desc.input],
            &desc.out,
        )
    }

    fn reduce_all(desc: &'a ReduceOpIr, reduction: ReductionKind) -> Self {
        let kind = OpKind::Reduction {
            dims: ReducedDims::All,
            kind: reduction,
        };
        Self::of_kind(kind, [&desc.input], &desc.out)
    }

    fn reduce_dim(desc: &'a ReduceDimOpIr, reduction: ReductionKind) -> Self {
        let kind = OpKind::Reduction {
            dims: ReducedDims::Dims(slice::from_ref(&desc.axis)),
            kind: reduction,
        };
        Self::of_kind(kind, [&desc.input], &desc.out)
    }

    fn matmul(desc: &'a MatmulOpIr) -> Self {
        Self::of_kind(OpKind::Matmul, [&desc.lhs, &desc.rhs], &desc.out)
    }

    /// The mask comes before the bias when the op has both.
    fn attention(desc: &'a AttentionOpIr) -> Self {
        let inputs = [&desc.query, &desc.key, &desc.value]
            .into_iter()
            .chain(desc.mask.as_ref())
            .chain(desc.attn_bias.as_ref())
            .collect();
        Self::Kind {
            kind: OpKind::Attention,
            inputs,
            out: &desc.out,
        }
    }

    fn of_kind<const N: usize>(
        kind: OpKind<'a>,
        inputs: [&'a TensorIr; N],
        out: &'a TensorIr,
    ) -> Self {
        Self::Kind {
            kind,
            inputs: inputs.into_iter().collect(),
            out,
        }
    }
}
