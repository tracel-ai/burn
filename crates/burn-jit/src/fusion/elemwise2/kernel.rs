use super::io::*;
use super::ir::*;
use cubecl::prelude::*;
use half::{bf16, f16};

#[cube]
pub fn fuse_on_write<E: CubePrimitive>(
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
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Div(op) => match op.out.precision() {
                OpPrecision::F32 => div::<f32>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::F16 => div::<f16>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::BF16 => {
                    div::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::I32 => div::<i32>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::U32 => div::<u32>(inputs, outputs, &mut locals, write_pos, op, config),
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Sub(op) => match op.out.precision() {
                OpPrecision::F32 => sub::<f32>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::F16 => sub::<f16>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::BF16 => {
                    sub::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::I32 => sub::<i32>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::U32 => sub::<u32>(inputs, outputs, &mut locals, write_pos, op, config),
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Mul(op) => match op.out.precision() {
                OpPrecision::F32 => mul::<f32>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::F16 => mul::<f16>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::BF16 => {
                    mul::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::I32 => mul::<i32>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::U32 => mul::<u32>(inputs, outputs, &mut locals, write_pos, op, config),
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Powf(op) => match op.out.precision() {
                OpPrecision::F32 => {
                    powf::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::F16 => {
                    powf::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::BF16 => {
                    powf::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Erf(op) => match op.out.precision() {
                OpPrecision::F32 => erf::<f32>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::F16 => erf::<f16>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::BF16 => {
                    erf::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Abs(op) => match op.out.precision() {
                OpPrecision::F32 => abs::<f32>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::F16 => abs::<f16>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::BF16 => {
                    abs::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Log(op) => match op.out.precision() {
                OpPrecision::F32 => log::<f32>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::F16 => log::<f16>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::BF16 => {
                    log::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Log1p(op) => match op.out.precision() {
                OpPrecision::F32 => {
                    log1p::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::F16 => {
                    log1p::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::BF16 => {
                    log1p::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Recip(op) => match op.out.precision() {
                OpPrecision::F32 => {
                    recip::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::F16 => {
                    recip::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::BF16 => {
                    recip::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Assign(op) => match op.out.precision() {
                OpPrecision::F32 => {
                    assign::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::F16 => {
                    assign::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::BF16 => {
                    assign::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::I32 => {
                    assign::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::U32 => {
                    assign::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::Bool => {
                    assign::<bool>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            _ => todo!(),
        }
    }
}

macro_rules! binary_op {
    ($ident:ident, $op:tt) => {
        #[cube]
        fn $ident<C: Numeric>(
            inputs: &FusionArgs,
            outputs: &mut FusionArgs,
            locals: &mut FusionLocals,
            write_pos: u32,
            #[comptime] op: BinaryElemwiseOp,
            #[comptime] config: &FusionConfig,
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
            inputs: &FusionArgs,
            outputs: &mut FusionArgs,
            locals: &mut FusionLocals,
            write_pos: u32,
            #[comptime] op: BinaryElemwiseOp,
            #[comptime] config: &FusionConfig,
        ) {
            let lhs = read::<C>(inputs, outputs, &locals, write_pos, op.lhs, config);
            let rhs = read::<C>(inputs, outputs, &locals, write_pos, op.rhs, config);
            let result = $func(lhs, rhs);

            write::<C>(inputs, outputs, locals, write_pos, result, op.out, config);
        }
    };
}

macro_rules! unary_func {
    ($ident:ident, $func:expr, $c:tt) => {
        #[cube]
        fn $ident<C: $c>(
            inputs: &FusionArgs,
            outputs: &mut FusionArgs,
            locals: &mut FusionLocals,
            write_pos: u32,
            #[comptime] op: UnaryElemwiseOp,
            #[comptime] config: &FusionConfig,
        ) {
            let input = read::<C>(inputs, outputs, &locals, write_pos, op.input, config);
            let result = $func(input);

            write::<C>(inputs, outputs, locals, write_pos, result, op.out, config);
        }
    };
}

#[cube]
fn assign<C: CubePrimitive>(
    inputs: &FusionArgs,
    outputs: &mut FusionArgs,
    locals: &mut FusionLocals,
    write_pos: u32,
    #[comptime] op: UnaryElemwiseOp,
    #[comptime] config: &FusionConfig,
) {
    let input = read::<C>(inputs, outputs, &locals, write_pos, op.input, config);

    write::<C>(inputs, outputs, locals, write_pos, input, op.out, config);
}

binary_op!(add, +);
binary_op!(mul, *);
binary_op!(div, /);
binary_op!(sub, -);

binary_func!(powf, Line::<C>::powf, Float);

unary_func!(exp, Line::<C>::exp, Float);
unary_func!(log, Line::<C>::log, Float);
unary_func!(log1p, Line::<C>::log1p, Float);
unary_func!(cos, Line::<C>::cos, Float);
unary_func!(sin, Line::<C>::sin, Float);
unary_func!(tanh, Line::<C>::tanh, Float);
unary_func!(erf, Line::<C>::erf, Float);
unary_func!(recip, Line::<C>::recip, Float);
unary_func!(abs, Line::<C>::abs, Float);
