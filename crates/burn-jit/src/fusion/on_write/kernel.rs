use super::io::*;
use super::ir::*;
use cubecl::prelude::*;
use half::{bf16, f16};

#[cube]
/// Fuse element-wise operations at the given write position.
///
/// You can start by writing some elements using `write_values` and `write_args`.
pub fn fuse_on_write<E: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    write_pos: u32,
    write_values: Registry<Arg, Line<E>>,
    #[comptime] write_args: Sequence<Arg>,
    #[comptime] config: &ElemwiseConfig,
) {
    let mut locals = LocalArgs {
        l_f32: Registry::<u32, Line<f32>>::new(),
        l_f16: Registry::<u32, Line<f16>>::new(),
        l_bf16: Registry::<u32, Line<bf16>>::new(),
        l_i64: Registry::<u32, Line<i64>>::new(),
        l_i32: Registry::<u32, Line<i32>>::new(),
        l_i16: Registry::<u32, Line<i16>>::new(),
        l_i8: Registry::<u32, Line<i8>>::new(),
        l_u64: Registry::<u32, Line<u64>>::new(),
        l_u32: Registry::<u32, Line<u32>>::new(),
        l_u16: Registry::<u32, Line<u16>>::new(),
        l_u8: Registry::<u32, Line<u8>>::new(),
        l_bool: Registry::<u32, Line<bool>>::new(),
    };

    // Write the values given as arguments.
    #[unroll]
    for i in 0..write_args.len() {
        let arg = comptime![write_args.index(i).clone()];
        let val = write_values.find(comptime![arg.clone()]);

        write::<E>(inputs, outputs, &mut locals, write_pos, val, arg, config);
    }

    #[unroll]
    for index in 0..config.ops.len() {
        let op = comptime! { config.ops.index(index).clone() };

        match op {
            ElemwiseOp::Add(op) => match op.out.precision() {
                ElemwisePrecision::F32 => {
                    add::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    add::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    add::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I64 => {
                    add::<i64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I32 => {
                    add::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I16 => {
                    add::<i16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I8 => {
                    add::<i8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U64 => {
                    add::<u64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U32 => {
                    add::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U16 => {
                    add::<u16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U8 => {
                    add::<u8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::Div(op) => match op.out.precision() {
                ElemwisePrecision::F32 => {
                    div::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    div::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    div::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I64 => {
                    div::<i64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I32 => {
                    div::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I16 => {
                    div::<i16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I8 => {
                    div::<i8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U64 => {
                    div::<u64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U32 => {
                    div::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U16 => {
                    div::<u16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U8 => {
                    div::<u8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::Sub(op) => match op.out.precision() {
                ElemwisePrecision::F32 => {
                    sub::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    sub::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    sub::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I64 => {
                    sub::<i64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I32 => {
                    sub::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I16 => {
                    sub::<i16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I8 => {
                    sub::<i8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U64 => {
                    sub::<u64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U32 => {
                    sub::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U16 => {
                    sub::<u16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U8 => {
                    sub::<u8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::Mul(op) => match op.out.precision() {
                ElemwisePrecision::F32 => {
                    mul::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    mul::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    mul::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I64 => {
                    mul::<i64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I32 => {
                    mul::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I16 => {
                    mul::<i16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I8 => {
                    mul::<i8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U64 => {
                    mul::<u64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U32 => {
                    mul::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U16 => {
                    mul::<u16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U8 => {
                    mul::<u8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::Powf(op) => match op.out.precision() {
                ElemwisePrecision::F32 => {
                    powf::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    powf::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    powf::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::Erf(op) => match op.out.precision() {
                ElemwisePrecision::F32 => {
                    erf::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    erf::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    erf::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::Abs(op) => match op.out.precision() {
                ElemwisePrecision::F32 => {
                    abs::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    abs::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    abs::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U64 => {
                    assign::<u64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U32 => {
                    assign::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U16 => {
                    assign::<u16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U8 => {
                    assign::<u8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I64 => {
                    abs::<i64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I32 => {
                    abs::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I16 => {
                    abs::<i16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I8 => {
                    abs::<i8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::Log(op) => match op.out.precision() {
                ElemwisePrecision::F32 => {
                    log::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    log::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    log::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::Log1p(op) => match op.out.precision() {
                ElemwisePrecision::F32 => {
                    log1p::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    log1p::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    log1p::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::Recip(op) => match op.out.precision() {
                ElemwisePrecision::F32 => {
                    recip::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    recip::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    recip::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::Assign(op) => match op.out.precision() {
                ElemwisePrecision::F32 => {
                    assign::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    assign::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    assign::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I64 => {
                    assign::<i64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I32 => {
                    assign::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I16 => {
                    assign::<i16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I8 => {
                    assign::<i8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U64 => {
                    assign::<u64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U32 => {
                    assign::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U16 => {
                    assign::<u16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U8 => {
                    assign::<u8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::Bool => {
                    assign::<bool>(inputs, outputs, &mut locals, write_pos, op, config)
                }
            },
            ElemwiseOp::Exp(op) => match op.out.precision() {
                ElemwisePrecision::F32 => {
                    exp::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    exp::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    exp::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::Cos(op) => match op.out.precision() {
                ElemwisePrecision::F32 => {
                    cos::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    cos::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    cos::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::Sin(op) => match op.out.precision() {
                ElemwisePrecision::F32 => {
                    sin::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    sin::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    sin::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::Tanh(op) => match op.out.precision() {
                ElemwisePrecision::F32 => {
                    tanh::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    tanh::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    tanh::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::Equal(op) => match op.lhs.precision() {
                ElemwisePrecision::F32 => {
                    equal::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    equal::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    equal::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I64 => {
                    equal::<i64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I32 => {
                    equal::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I16 => {
                    equal::<i16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I8 => {
                    equal::<i8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U64 => {
                    equal::<u64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U32 => {
                    equal::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U16 => {
                    equal::<u16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U8 => {
                    equal::<u8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::Bool => {
                    equal::<bool>(inputs, outputs, &mut locals, write_pos, op, config)
                }
            },
            ElemwiseOp::Greater(op) => match op.lhs.precision() {
                ElemwisePrecision::F32 => {
                    greater::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    greater::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    greater::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I64 => {
                    greater::<i64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I32 => {
                    greater::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I16 => {
                    greater::<i16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I8 => {
                    greater::<i8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U64 => {
                    greater::<u64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U32 => {
                    greater::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U16 => {
                    greater::<u16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U8 => {
                    greater::<u8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::GreaterEqual(op) => match op.lhs.precision() {
                ElemwisePrecision::F32 => {
                    greater_equal::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    greater_equal::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    greater_equal::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I64 => {
                    greater_equal::<i64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I32 => {
                    greater_equal::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I16 => {
                    greater_equal::<i16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I8 => {
                    greater_equal::<i8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U64 => {
                    greater_equal::<u64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U32 => {
                    greater_equal::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U16 => {
                    greater_equal::<u16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U8 => {
                    greater_equal::<u8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::Lower(op) => match op.lhs.precision() {
                ElemwisePrecision::F32 => {
                    lower::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    lower::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    lower::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I64 => {
                    lower::<i64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I32 => {
                    lower::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I16 => {
                    lower::<i16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I8 => {
                    lower::<i8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U64 => {
                    lower::<u64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U32 => {
                    lower::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U16 => {
                    lower::<u16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U8 => {
                    lower::<u8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::LowerEqual(op) => match op.lhs.precision() {
                ElemwisePrecision::F32 => {
                    lower_equal::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::F16 => {
                    lower_equal::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::BF16 => {
                    lower_equal::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I64 => {
                    lower_equal::<i64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I32 => {
                    lower_equal::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I16 => {
                    lower_equal::<i16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::I8 => {
                    lower_equal::<i8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U64 => {
                    lower_equal::<u64>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U32 => {
                    lower_equal::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U16 => {
                    lower_equal::<u16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                ElemwisePrecision::U8 => {
                    lower_equal::<u8>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported precision {op:?}")],
            },
            ElemwiseOp::ConditionalAssign {
                cond,
                lhs,
                rhs,
                out,
            } => match out.precision() {
                ElemwisePrecision::F32 => conditional_assign::<f32>(
                    inputs,
                    outputs,
                    &mut locals,
                    write_pos,
                    cond,
                    lhs,
                    rhs,
                    out,
                    config,
                ),
                ElemwisePrecision::F16 => conditional_assign::<f16>(
                    inputs,
                    outputs,
                    &mut locals,
                    write_pos,
                    cond,
                    lhs,
                    rhs,
                    out,
                    config,
                ),
                ElemwisePrecision::BF16 => conditional_assign::<bf16>(
                    inputs,
                    outputs,
                    &mut locals,
                    write_pos,
                    cond,
                    lhs,
                    rhs,
                    out,
                    config,
                ),
                ElemwisePrecision::I64 => conditional_assign::<i64>(
                    inputs,
                    outputs,
                    &mut locals,
                    write_pos,
                    cond,
                    lhs,
                    rhs,
                    out,
                    config,
                ),
                ElemwisePrecision::I32 => conditional_assign::<i32>(
                    inputs,
                    outputs,
                    &mut locals,
                    write_pos,
                    cond,
                    lhs,
                    rhs,
                    out,
                    config,
                ),
                ElemwisePrecision::I16 => conditional_assign::<i16>(
                    inputs,
                    outputs,
                    &mut locals,
                    write_pos,
                    cond,
                    lhs,
                    rhs,
                    out,
                    config,
                ),
                ElemwisePrecision::I8 => conditional_assign::<i8>(
                    inputs,
                    outputs,
                    &mut locals,
                    write_pos,
                    cond,
                    lhs,
                    rhs,
                    out,
                    config,
                ),
                ElemwisePrecision::U64 => conditional_assign::<u64>(
                    inputs,
                    outputs,
                    &mut locals,
                    write_pos,
                    cond,
                    lhs,
                    rhs,
                    out,
                    config,
                ),
                ElemwisePrecision::U32 => conditional_assign::<u32>(
                    inputs,
                    outputs,
                    &mut locals,
                    write_pos,
                    cond,
                    lhs,
                    rhs,
                    out,
                    config,
                ),
                ElemwisePrecision::U16 => conditional_assign::<u16>(
                    inputs,
                    outputs,
                    &mut locals,
                    write_pos,
                    cond,
                    lhs,
                    rhs,
                    out,
                    config,
                ),
                ElemwisePrecision::U8 => conditional_assign::<u8>(
                    inputs,
                    outputs,
                    &mut locals,
                    write_pos,
                    cond,
                    lhs,
                    rhs,
                    out,
                    config,
                ),
                _ => comptime![panic!("Unsupported precision")],
            },
        }
    }
}

macro_rules! binary_op {
    ($ident:ident, $op:tt) => {
        #[cube]
        fn $ident<C: Numeric>(
            inputs: &GlobalArgs,
            outputs: &mut GlobalArgs,
            locals: &mut LocalArgs,
            write_pos: u32,
            #[comptime] op: BinaryElemwiseArgs,
            #[comptime] config: &ElemwiseConfig,
        ) {
            let lhs = read::<C>(inputs, outputs, &locals, write_pos, op.lhs, config);
            let rhs = read::<C>(inputs, outputs, &locals, write_pos, op.rhs, config);
            let result = lhs $op rhs;

            write::<C>(inputs, outputs, locals, write_pos, result, op.out, config);
        }
    };
}

macro_rules! binary_func {
    ($ident:ident, $func:expr, $c:tt) => {
        #[cube]
        fn $ident<C: $c>(
            inputs: &GlobalArgs,
            outputs: &mut GlobalArgs,
            locals: &mut LocalArgs,
            write_pos: u32,
            #[comptime] op: BinaryElemwiseArgs,
            #[comptime] config: &ElemwiseConfig,
        ) {
            let lhs = read::<C>(inputs, outputs, &locals, write_pos, op.lhs, config);
            let rhs = read::<C>(inputs, outputs, &locals, write_pos, op.rhs, config);
            let result = $func(lhs, rhs);

            write::<C>(inputs, outputs, locals, write_pos, result, op.out, config);
        }
    };
}

macro_rules! comparison_op {
    ($ident:ident, $op:tt) => {
        #[cube]
        fn $ident<C: CubePrimitive + core::cmp::PartialOrd>(
            inputs: &GlobalArgs,
            outputs: &mut GlobalArgs,
            locals: &mut LocalArgs,
            write_pos: u32,
            #[comptime] op: BinaryElemwiseArgs,
            #[comptime] config: &ElemwiseConfig,
        ) {
            let lhs = read::<C>(inputs, outputs, &locals, write_pos, op.lhs, config);
            let rhs = read::<C>(inputs, outputs, &locals, write_pos, op.rhs, config);
            let result = Line::new(lhs $op rhs);

            write::<bool>(inputs, outputs, locals, write_pos, result, op.out, config);
        }
    };
}

macro_rules! unary_func {
    ($ident:ident, $func:expr, $c:tt) => {
        #[cube]
        fn $ident<C: $c>(
            inputs: &GlobalArgs,
            outputs: &mut GlobalArgs,
            locals: &mut LocalArgs,
            write_pos: u32,
            #[comptime] op: UnaryElemwiseArgs,
            #[comptime] config: &ElemwiseConfig,
        ) {
            let input = read::<C>(inputs, outputs, &locals, write_pos, op.input, config);
            let result = $func(input);

            write::<C>(inputs, outputs, locals, write_pos, result, op.out, config);
        }
    };
}

#[cube]
fn assign<C: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    write_pos: u32,
    #[comptime] op: UnaryElemwiseArgs,
    #[comptime] config: &ElemwiseConfig,
) {
    let input = read::<C>(inputs, outputs, locals, write_pos, op.input, config);

    write::<C>(inputs, outputs, locals, write_pos, input, op.out, config);
}

#[cube]
fn conditional_assign<C: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    write_pos: u32,
    #[comptime] cond: Arg,
    #[comptime] lhs: Arg,
    #[comptime] rhs: Arg,
    #[comptime] out: Arg,
    #[comptime] config: &ElemwiseConfig,
) {
    let cond = read::<bool>(inputs, outputs, locals, write_pos, cond, config);
    let lhs = read::<C>(inputs, outputs, locals, write_pos, lhs, config);
    let rhs = read::<C>(inputs, outputs, locals, write_pos, rhs, config);
    let result = select_many(cond, lhs, rhs);

    write::<C>(inputs, outputs, locals, write_pos, result, out, config);
}

binary_op!(add, +);
binary_op!(mul, *);
binary_op!(div, /);
binary_op!(sub, -);

comparison_op!(equal, ==);
comparison_op!(greater, >);
comparison_op!(greater_equal, >=);
comparison_op!(lower, <);
comparison_op!(lower_equal, <=);

binary_func!(powf, Line::<C>::powf, Float);

unary_func!(exp, Line::<C>::exp, Float);
unary_func!(log, Line::<C>::log, Float);
unary_func!(log1p, Line::<C>::log1p, Float);
unary_func!(cos, Line::<C>::cos, Float);
unary_func!(sin, Line::<C>::sin, Float);
unary_func!(tanh, Line::<C>::tanh, Float);
unary_func!(erf, Line::<C>::erf, Float);
unary_func!(recip, Line::<C>::recip, Float);
unary_func!(abs, Line::<C>::abs, Numeric);
