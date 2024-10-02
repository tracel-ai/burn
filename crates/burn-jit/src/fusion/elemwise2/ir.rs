use burn_tensor::DType;
use cubecl::linalg::tensor::index_offset_with_layout;
pub use cubecl::prelude::*;
use half::f16;

#[derive(CubeType, Clone, Copy)]
pub enum Arg {
    Input(u32),
    Local(u32),
    Output(u32),
    Scalar(u32),
}

#[derive(CubeType)]
pub enum ElemwiseOp {
    Add(BinaryElemwiseOp),
    Sub(BinaryElemwiseOp),
    Mul(BinaryElemwiseOp),
    Pow(UnaryElemwiseOp),
    Assign(UnaryElemwiseOp),
    ToLayout(UnaryElemwiseOp),
}

#[derive(CubeLaunch)]
pub struct FusionArgs {
    t_f32: Sequence<Tensor<Line<f32>>>,
    t_f16: Sequence<Tensor<Line<f16>>>,
    t_i32: Sequence<Tensor<Line<i32>>>,
    t_u32: Sequence<Tensor<Line<u32>>>,
    s_f32: Sequence<f32>,
    s_f16: Sequence<f16>,
    s_i32: Sequence<i32>,
    s_u32: Sequence<u32>,
}

#[derive(CubeType)]
pub struct FusionLocals {
    l_f32: Sequence<Line<f32>>,
    l_f16: Sequence<Line<f16>>,
    l_i32: Sequence<Line<i32>>,
    l_u32: Sequence<Line<u32>>,
    l_bool: Sequence<Line<bool>>,
}

#[derive(CubeType)]
pub struct UnaryElemwiseOp {
    pub input: Arg,
    pub out: Arg,
    pub precision: OpPrecision,
}

#[derive(CubeType)]
pub struct BinaryElemwiseOp {
    pub lhs: Arg,
    pub rhs: Arg,
    pub out: Arg,
    pub precision: OpPrecision,
}

#[derive(CubeType, Clone, Copy)]
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

#[derive(CubeType, Clone, Copy)]
pub struct RefLayout {
    pub precision: OpPrecision,
    pub arg: Arg,
}

#[cube]
pub fn get_offset<C: CubePrimitive>(
    inputs: &FusionArgs,
    outputs: &FusionArgs,
    tensor: &Tensor<Line<C>>,
    pos: u32,
    #[comptime] rank: u32,
    #[comptime] ref_layout: RefLayout,
) -> u32 {
    match comptime![ref_layout.precision] {
        OpPrecision::F32 => match comptime![ref_layout.arg] {
            Arg::Input(index) => {
                let layout = inputs.t_f32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, rank, true)
            }
            Arg::Output(index) => {
                let layout = outputs.t_f32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, rank, true)
            }
            _ => todo!(),
        },
        _ => todo!(),
    }
}

#[cube]
pub fn read_f16(
    inputs: &FusionArgs,
    outputs: &FusionArgs,
    locals: &FusionLocals,
    ref_pos: u32,
    #[comptime] ref_layout: RefLayout,
    #[comptime] arg: Arg,
    #[comptime] rank: u32,
) -> Line<f16> {
    match arg {
        Arg::Input(index) => {
            let tensor = inputs.t_f16.index(index);
            let offset = get_offset(inputs, outputs, tensor, ref_pos, rank, ref_layout);
            tensor[offset]
        }
        Arg::Local(index) => *locals.l_f16.index(index),
        _ => comptime![panic!("Invalid")],
    }
}

#[cube]
pub fn write_f16(
    inputs: &FusionArgs,
    outputs: &mut FusionArgs,
    locals: &FusionLocals,
    ref_pos: u32,
    #[comptime] ref_layout: RefLayout,
    #[comptime] arg: Arg,
    #[comptime] rank: u32,
    value: Line<f16>,
) {
    match arg {
        Arg::Input(index) => {
            let offset = {
                let tensor = outputs.t_f16.index(index);
                get_offset(inputs, outputs, tensor, ref_pos, rank, ref_layout)
            };

            let output = outputs.t_f16.index_mut(index);
            output[offset] = value;
        }
        _ => comptime![panic!("Invalid")],
    }
}

#[cube]
fn fuse(
    inputs: &FusionArgs,
    outputs: &mut FusionArgs,
    ref_pos: u32,
    #[comptime] ref_layout: RefLayout,
    #[comptime] ops: Sequence<ElemwiseOp>,
    #[comptime] rank: u32,
) {
    let mut locals = FusionLocals {
        l_f32: Sequence::new(),
        l_f16: Sequence::new(),
        l_i32: Sequence::new(),
        l_u32: Sequence::new(),
        l_bool: Sequence::new(),
    };

    #[unroll]
    for index in 0..ops.len() {
        let op = comptime! { ops.index(index).clone() };

        match op {
            ElemwiseOp::Add(op) => match op.precision {
                OpPrecision::F16 => {
                    let lhs = read_f16(inputs, outputs, &locals, ref_pos, ref_layout, op.lhs, rank);
                    let rhs = read_f16(inputs, outputs, &locals, ref_pos, ref_layout, op.rhs, rank);
                    let result = lhs + rhs;

                    write_f16(inputs, outputs, &locals, ref_pos, ref_layout, op.rhs, rank, result);
                }
                _ => unsupported(None),
            },
            ElemwiseOp::Sub(_) => todo!(),
            ElemwiseOp::Mul(_) => todo!(),
            ElemwiseOp::Pow(_) => todo!(),
            ElemwiseOp::Assign(_) => todo!(),
            ElemwiseOp::ToLayout(_) => todo!(),
        }
    }
}
