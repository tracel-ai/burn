use burn_tensor::DType;
use cubecl::ir::Elem;
use cubecl::prelude::*;
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

#[derive(
    CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord,
)]
/// Argument to an [elemwise operation](ElemwiseOp).
pub enum Arg {
    Input(u32, ElemwisePrecision, LayoutInfo),
    Local(u32, ElemwisePrecision),
    Output(u32, ElemwisePrecision, LayoutInfo),
    Scalar(u32, ElemwisePrecision),
    /// Only constant that can be encoded into an u32 can be used as literal.
    Literal(u32, ElemwisePrecision),
}

#[derive(
    CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord,
)]
/// Layout information.
pub enum LayoutInfo {
    /// The layout if the same as the reference.
    SameAsRef,
    /// The reference layout.
    IsRef,
    /// The layout if unknown.
    Unknown,
}

impl Arg {
    pub fn precision(&self) -> ElemwisePrecision {
        *match self {
            Arg::Input(_, p, _) => p,
            Arg::Local(_, p) => p,
            Arg::Output(_, p, _) => p,
            Arg::Scalar(_, p) => p,
            Arg::Literal(_, p) => p,
        }
    }
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// Operations that can be executed and fused.
pub enum ElemwiseOp {
    Add(BinaryElemwiseArgs),
    Sub(BinaryElemwiseArgs),
    Mul(BinaryElemwiseArgs),
    Div(BinaryElemwiseArgs),
    Powf(BinaryElemwiseArgs),
    Abs(UnaryElemwiseArgs),
    Exp(UnaryElemwiseArgs),
    Log(UnaryElemwiseArgs),
    Log1p(UnaryElemwiseArgs),
    Cos(UnaryElemwiseArgs),
    Sin(UnaryElemwiseArgs),
    Tanh(UnaryElemwiseArgs),
    Erf(UnaryElemwiseArgs),
    Recip(UnaryElemwiseArgs),
    Assign(UnaryElemwiseArgs),
    Equal(BinaryElemwiseArgs),
    Lower(BinaryElemwiseArgs),
    Greater(BinaryElemwiseArgs),
    LowerEqual(BinaryElemwiseArgs),
    GreaterEqual(BinaryElemwiseArgs),
    ConditionalAssign {
        cond: Arg,
        lhs: Arg,
        rhs: Arg,
        out: Arg,
    },
}

#[derive(CubeLaunch)]
/// Global arguments that are used for fusing [element wise operations](ElemwiseOp).
pub struct GlobalArgs {
    pub t_f32: Sequence<Tensor<Line<f32>>>,
    pub t_f16: Sequence<Tensor<Line<f16>>>,
    pub t_bf16: Sequence<Tensor<Line<bf16>>>,
    pub t_i32: Sequence<Tensor<Line<i32>>>,
    pub t_u32: Sequence<Tensor<Line<u32>>>,
    pub s_f32: Sequence<f32>,
    pub s_f16: Sequence<f16>,
    pub s_bf16: Sequence<bf16>,
    pub s_i32: Sequence<i32>,
    pub s_u32: Sequence<u32>,
}

#[derive(CubeType, Clone)]
/// Keep track of all local variables that are used as argument in fused
/// [element wise operations](ElemwiseOp).
pub struct LocalArgs {
    pub l_f32: Registry<u32, Line<f32>>,
    pub l_f16: Registry<u32, Line<f16>>,
    pub l_bf16: Registry<u32, Line<bf16>>,
    pub l_i32: Registry<u32, Line<i32>>,
    pub l_u32: Registry<u32, Line<u32>>,
    pub l_bool: Registry<u32, Line<bool>>,
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// Unary [element wise operation](ElemwiseOp) arguments.
pub struct UnaryElemwiseArgs {
    pub input: Arg,
    pub out: Arg,
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// Binary [element wise operation](ElemwiseOp) arguments.
pub struct BinaryElemwiseArgs {
    pub lhs: Arg,
    pub rhs: Arg,
    pub out: Arg,
}

#[derive(
    CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
/// Precisions supported by [element wise operations](ElemwiseOp).
pub enum ElemwisePrecision {
    F32,
    F16,
    BF16,
    I32,
    I8,
    U32,
    U8,
    Bool,
}

impl From<Elem> for ElemwisePrecision {
    fn from(value: Elem) -> Self {
        match value {
            Elem::Float(kind) => match kind {
                cubecl::ir::FloatKind::F16 => Self::F16,
                cubecl::ir::FloatKind::BF16 => Self::BF16,
                cubecl::ir::FloatKind::F32 => Self::F32,
                _ => panic!("Unsupported precision for fusion: {value}"),
            },
            Elem::Int(cubecl::ir::IntKind::I32) => Self::I32,
            Elem::UInt => Self::U32,
            Elem::Bool => Self::Bool,
            _ => panic!("Unsupported precision for fusion: {value}"),
        }
    }
}

impl From<DType> for ElemwisePrecision {
    fn from(value: DType) -> Self {
        match value {
            DType::F32 => Self::F32,
            DType::F16 => Self::F16,
            DType::BF16 => Self::BF16,
            DType::I32 => Self::I32,
            DType::I8 => Self::I8,
            DType::U32 => Self::U32,
            DType::U8 => Self::U8,
            DType::Bool => Self::Bool,
            _ => panic!("Unsupported"),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
/// Configuration that encapsulates all comptime information necessary for element wise fusion.
pub struct ElemwiseConfig {
    pub rank: u32,
    pub ref_layout: Arg,
    pub ops: Sequence<ElemwiseOp>,
}

impl Arg {
    /// Add layout information; it's going to impact how the input or output is read
    /// and written to.
    pub fn add_layout_info(&mut self, layout: LayoutInfo) {
        match self {
            Arg::Input(_, _, old) => {
                *old = layout;
            }
            Arg::Output(_, _, old) => {
                *old = layout;
            }
            _ => {}
        }
    }
}

impl RegistryQuery<Self> for Arg {}
