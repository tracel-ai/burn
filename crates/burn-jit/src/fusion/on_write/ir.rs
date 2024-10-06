use burn_tensor::DType;
use cubecl::ir::Elem;
use cubecl::prelude::*;
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

#[derive(
    CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord,
)]
pub enum Arg {
    Input(u32, OpPrecision, LayoutInfo),
    Local(u32, OpPrecision),
    Output(u32, OpPrecision, LayoutInfo),
    Scalar(u32, OpPrecision),
    /// Only constant that can be encoded into an u32 can be used as literal.
    Literal(u32, OpPrecision),
}

impl Arg {
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

impl ComptimeRegistryQuery<Self> for Arg {}

#[derive(
    CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord,
)]
pub enum LayoutInfo {
    SameAsRef,
    IsRef,
    Unknown,
}

impl Arg {
    pub fn precision(&self) -> OpPrecision {
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
pub enum ElemwiseOp {
    Add(BinaryElemwiseOp),
    Sub(BinaryElemwiseOp),
    Mul(BinaryElemwiseOp),
    Div(BinaryElemwiseOp),
    Powf(BinaryElemwiseOp),
    Abs(UnaryElemwiseOp),
    Exp(UnaryElemwiseOp),
    Log(UnaryElemwiseOp),
    Log1p(UnaryElemwiseOp),
    Cos(UnaryElemwiseOp),
    Sin(UnaryElemwiseOp),
    Tanh(UnaryElemwiseOp),
    Erf(UnaryElemwiseOp),
    Recip(UnaryElemwiseOp),
    Assign(UnaryElemwiseOp),
    ConditionalAssign {
        cond: Arg,
        lhs: Arg,
        rhs: Arg,
        out: Arg,
    },
    Equal(BinaryElemwiseOp),
    Lower(BinaryElemwiseOp),
    Greater(BinaryElemwiseOp),
    LowerEqual(BinaryElemwiseOp),
    GreaterEqual(BinaryElemwiseOp),
}

#[derive(CubeLaunch)]
pub struct FusionArgs {
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
pub struct FusionLocals {
    pub l_f32: ComptimeRegistry<u32, Line<f32>>,
    pub l_f16: ComptimeRegistry<u32, Line<f16>>,
    pub l_bf16: ComptimeRegistry<u32, Line<bf16>>,
    pub l_i32: ComptimeRegistry<u32, Line<i32>>,
    pub l_u32: ComptimeRegistry<u32, Line<u32>>,
    pub l_bool: ComptimeRegistry<u32, Line<bool>>,
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnaryElemwiseOp {
    pub input: Arg,
    pub out: Arg,
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinaryElemwiseOp {
    pub lhs: Arg,
    pub rhs: Arg,
    pub out: Arg,
}

#[derive(
    CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum OpPrecision {
    F32,
    F16,
    BF16,
    I32,
    I8,
    U32,
    U8,
    Bool,
}

impl From<Elem> for OpPrecision {
    fn from(value: Elem) -> Self {
        match value {
            Elem::Float(kind) => match kind {
                cubecl::ir::FloatKind::F16 => Self::F16,
                cubecl::ir::FloatKind::BF16 => Self::BF16,
                cubecl::ir::FloatKind::F32 => Self::F32,
                _ => panic!("Unsupported precision for fusion: {value}"),
            },
            Elem::Int(kind) => match kind {
                cubecl::ir::IntKind::I32 => Self::I32,
                _ => panic!("Unsupported precision for fusion: {value}"),
            },
            Elem::UInt => Self::U32,
            Elem::Bool => Self::Bool,
            _ => panic!("Unsupported precision for fusion: {value}"),
        }
    }
}

impl From<DType> for OpPrecision {
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

#[derive(CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct RefLayout {
    pub arg: Arg,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct FusionConfig {
    pub rank: u32,
    pub ref_layout: RefLayout,
    pub ops: Sequence<ElemwiseOp>,
}
