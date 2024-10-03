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
        l_bf16: Sequence::new(),
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
            ElemwiseOp::Exp(op) => match op.out.precision() {
                OpPrecision::F32 => exp::<f32>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::F16 => exp::<f16>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::BF16 => {
                    exp::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Cos(op) => match op.out.precision() {
                OpPrecision::F32 => cos::<f32>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::F16 => cos::<f16>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::BF16 => {
                    cos::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Sin(op) => match op.out.precision() {
                OpPrecision::F32 => sin::<f32>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::F16 => sin::<f16>(inputs, outputs, &mut locals, write_pos, op, config),
                OpPrecision::BF16 => {
                    sin::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Tanh(op) => match op.out.precision() {
                OpPrecision::F32 => {
                    tanh::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::F16 => {
                    tanh::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::BF16 => {
                    tanh::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Equal(op) => match op.out.precision() {
                OpPrecision::F32 => {
                    equal::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::F16 => {
                    equal::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::BF16 => {
                    equal::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::I32 => {
                    equal::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::U32 => {
                    equal::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Greater(op) => match op.out.precision() {
                OpPrecision::F32 => {
                    greater::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::F16 => {
                    greater::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::BF16 => {
                    greater::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::I32 => {
                    greater::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::U32 => {
                    greater::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::GreaterEqual(op) => match op.out.precision() {
                OpPrecision::F32 => {
                    greater_equal::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::F16 => {
                    greater_equal::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::BF16 => {
                    greater_equal::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::I32 => {
                    greater_equal::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::U32 => {
                    greater_equal::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::Lower(op) => match op.out.precision() {
                OpPrecision::F32 => {
                    lower::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::F16 => {
                    lower::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::BF16 => {
                    lower::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::I32 => {
                    lower::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::U32 => {
                    lower::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::LowerEqual(op) => match op.out.precision() {
                OpPrecision::F32 => {
                    lower_equal::<f32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::F16 => {
                    lower_equal::<f16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::BF16 => {
                    lower_equal::<bf16>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::I32 => {
                    lower_equal::<i32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                OpPrecision::U32 => {
                    lower_equal::<u32>(inputs, outputs, &mut locals, write_pos, op, config)
                }
                _ => comptime![panic!("Unsupported")],
            },
            ElemwiseOp::ConditionalAssign {
                cond,
                lhs,
                rhs,
                out,
            } => match out.precision() {
                OpPrecision::F32 => conditional_assign::<f32>(
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
                OpPrecision::F16 => conditional_assign::<f16>(
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
                OpPrecision::BF16 => conditional_assign::<bf16>(
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
                OpPrecision::I32 => conditional_assign::<i32>(
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
                OpPrecision::U32 => conditional_assign::<u32>(
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
                _ => comptime![panic!("Unsupported")],
            },
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

macro_rules! comparison_op {
    ($ident:ident, $op:tt) => {
        #[cube]
        fn $ident<C: CubePrimitive + core::cmp::PartialOrd>(
            inputs: &FusionArgs,
            outputs: &mut FusionArgs,
            locals: &mut FusionLocals,
            write_pos: u32,
            #[comptime] op: BinaryElemwiseOp,
            #[comptime] config: &FusionConfig,
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

#[cube]
fn conditional_assign<C: CubePrimitive>(
    inputs: &FusionArgs,
    outputs: &mut FusionArgs,
    locals: &mut FusionLocals,
    write_pos: u32,
    #[comptime] cond: Arg,
    #[comptime] lhs: Arg,
    #[comptime] rhs: Arg,
    #[comptime] out: Arg,
    #[comptime] config: &FusionConfig,
) {
    let cond = read::<bool>(inputs, outputs, &locals, write_pos, cond, config);
    let lhs = read::<C>(inputs, outputs, &locals, write_pos, lhs, config);
    let rhs = read::<C>(inputs, outputs, &locals, write_pos, rhs, config);
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
unary_func!(abs, Line::<C>::abs, Float);
