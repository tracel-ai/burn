use super::io::*;
use burn_tensor::DType;
pub use cubecl::prelude::*;
use cubecl::{ir::Elem, linalg::tensor::index_offset_with_layout};
use half::{bf16, f16};

#[derive(CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Arg {
    Input(u32, OpPrecision),
    Local(u32, OpPrecision),
    Output(u32, OpPrecision),
    Scalar(u32, OpPrecision),
    /// Only constant that can be encoded into an u32 can be used as literal.
    Literal(u32, OpPrecision),
}

impl Arg {
    pub fn precision(&self) -> OpPrecision {
        *match self {
            Arg::Input(_, p) => p,
            Arg::Local(_, p) => p,
            Arg::Output(_, p) => p,
            Arg::Scalar(_, p) => p,
            Arg::Literal(_, p) => p,
        }
    }
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq)]
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
    ToLayout(UnaryElemwiseOp),
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
    pub t_i32: Sequence<Tensor<Line<i32>>>,
    pub t_u32: Sequence<Tensor<Line<u32>>>,
    pub s_f32: Sequence<f32>,
    pub s_f16: Sequence<f16>,
    pub s_i32: Sequence<i32>,
    pub s_u32: Sequence<u32>,
}

#[derive(CubeType, Clone)]
pub struct FusionLocals {
    pub l_f32: Sequence<Line<f32>>,
    pub l_f16: Sequence<Line<f16>>,
    pub l_i32: Sequence<Line<i32>>,
    pub l_u32: Sequence<Line<u32>>,
    pub l_bool: Sequence<Line<bool>>,
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnaryElemwiseOp {
    pub input: Arg,
    pub out: Arg,
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq)]
pub struct BinaryElemwiseOp {
    pub lhs: Arg,
    pub rhs: Arg,
    pub out: Arg,
}

#[derive(CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum OpPrecision {
    F32,
    F16,
    BF16,
    I32,
    I16,
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
            DType::I16 => Self::I16,
            DType::I8 => Self::I8,
            DType::U32 => Self::U32,
            DType::U8 => Self::U8,
            DType::Bool => Self::Bool,
            _ => panic!("Unsupported"),
        }
    }
}

#[cube]
fn unsupported(#[comptime] message: Option<&str>) {
    match message {
        Some(msg) => panic!("{msg}"),
        None => panic!("Unsupported"),
    }
}

#[derive(CubeType)]
pub enum ReadPosition {
    ToLayout { ref_pos: u32, ref_layout: RefLayout },
    Plain { pos: u32 },
    Unspecified,
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

#[cube]
fn fuse_on_write<E: CubePrimitive>(
    inputs: &FusionArgs,
    outputs: &mut FusionArgs,
    write_pos: u32,
    write_value: Line<E>,
    #[comptime] write_arg: Option<Arg>,
    #[comptime] config: &FusionConfig,
) {
    let mut locals = FusionLocals {
        l_f32: Sequence::new(),
        l_f16: Sequence::new(),
        l_i32: Sequence::new(),
        l_u32: Sequence::new(),
        l_bool: Sequence::new(),
    };

    // Initialize the write value.
    match write_arg {
        Some(val) => {
            write::<E>(
                inputs,
                outputs,
                &mut locals,
                write_pos,
                write_value,
                val,
                config,
            );
        }
        None => {}
    };

    #[unroll]
    for index in 0..config.ops.len() {
        let op = comptime! { config.ops.index(index).clone() };

        match op {
            ElemwiseOp::Add(op) => match op.out.precision() {
                OpPrecision::F32 => add::<f32>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::F16 => add::<f16>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::BF16 => {
                    add::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::I32 => add::<i32>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::U32 => add::<u32>(inputs, outputs, &mut locals, write_pos, op, config),
                _ => unsupported(None),
            },
            ElemwiseOp::Sub(_) => todo!(),
            ElemwiseOp::Mul(_) => todo!(),
            ElemwiseOp::Powf(_) => todo!(),
            ElemwiseOp::Assign(_) => todo!(),
            ElemwiseOp::ToLayout(_) => todo!(),
            _ => todo!(),
        }
    }
}

#[cube]
fn add<C: Numeric>(
    inputs: &FusionArgs,
    outputs: &mut FusionArgs,
    locals: &mut FusionLocals,
    write_pos: u32,
    #[comptime] op: BinaryElemwiseOp,
    #[comptime] config: &FusionConfig,
) {
    let lhs = read::<C>(inputs, outputs, &locals, write_pos, op.lhs, config);
    let rhs = read::<C>(inputs, outputs, &locals, write_pos, op.rhs, config);
    let result = lhs + rhs;

    write::<C>(inputs, outputs, locals, write_pos, result, op.rhs, config);
}

#[cube(launch_unchecked)]
fn elemwise_fuse(inputs: &FusionArgs, outputs: &mut FusionArgs, #[comptime] config: &FusionConfig) {
    fuse_on_write::<f32>(inputs, outputs, ABSOLUTE_POS, Line::empty(1), None, config)
}
