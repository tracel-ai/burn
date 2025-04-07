use crate::shared::DYN_ELEM_ID;

use super::io::*;
use super::ir::*;
use cubecl::prelude::*;

#[cube]
/// Fuse element-wise operations at the given write position.
///
/// You can start by writing some elements using `write_values` and `write_args`.
pub fn fuse_on_write<E: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    write_pos: u32,
    write_values: Registry<Arg, Line<E>>,
    #[comptime] write_args: Sequence<Arg>,
    #[comptime] config: &FuseBlockConfig,
) {
    // Write the values given as arguments.
    #[unroll]
    for i in 0..write_args.len() {
        let arg = comptime![write_args.index(i).clone()];
        let val = write_values.find(comptime![arg.clone()]);

        write::<E>(inputs, outputs, locals, write_pos, val, arg, config);
    }

    fuse(inputs, outputs, locals, write_pos, config);
}

#[cube]
/// Fuse element-wise operations at the given read position.
pub fn fuse_on_read<E: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    read_pos: u32,
    #[comptime] read_args: Sequence<Arg>,
    #[comptime] config: &FuseBlockConfig,
) -> Sequence<Line<E>> {
    fuse(inputs, outputs, locals, read_pos, config);

    let mut output = Sequence::new();

    #[unroll]
    for i in 0..read_args.len() {
        let arg = comptime![read_args.index(i).clone()];
        let value = read::<E>(inputs, outputs, locals, read_pos, arg, config);

        output.push(value);
    }

    output
}

#[cube]
pub fn init_locals(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    #[comptime] config: &FuseBlockConfig,
) -> LocalArgs {
    let mut ref_shape = Array::new(config.rank);
    let mut ref_strides = Array::new(config.rank);

    match comptime![config.ref_layout.clone()] {
        RefLayout::Concrete(arg) => match comptime![arg] {
            Arg::Input(index, ..) => {
                let layout = inputs.tensors.index(index);

                #[unroll]
                for i in 0..config.rank {
                    ref_shape[i] = layout.tensor.shape(i);
                    ref_strides[i] = layout.tensor.stride(i);
                }

                LocalArgs::new(
                    ref_shape.to_slice(),
                    ref_strides.to_slice(),
                    layout.tensor.line_size(),
                )
            }
            Arg::Output(index, ..) => {
                let layout = outputs.tensors.index(index);

                #[unroll]
                for i in 0..config.rank {
                    ref_shape[i] = layout.tensor.shape(i);
                    ref_strides[i] = layout.tensor.stride(i);
                }

                LocalArgs::new(
                    ref_shape.to_slice(),
                    ref_strides.to_slice(),
                    layout.tensor.line_size(),
                )
            }
            _ => comptime![panic!("Invalid concrete ref layout.")],
        },
        RefLayout::Virtual(layout) => match layout {
            VirtualLayout::SwapDims(original, dims) => {
                let layout = match comptime![original.clone()] {
                    Arg::Input(pos, ..) => inputs.tensors.index(pos),
                    Arg::Output(pos, ..) => outputs.tensors.index(pos),
                    _ => comptime![panic!("Unsupported")],
                };

                let mut stride_curr = 1u32;

                #[unroll]
                #[allow(clippy::clone_on_copy)]
                for i in 0..config.rank {
                    let reverse = comptime![reverse_index(config.rank, comptime![i.clone()])];
                    let swap = comptime![swap_dims_transform(comptime![&reverse], dims)];
                    let shape = layout.tensor.shape(comptime![swap.clone()]);

                    ref_shape[comptime![reverse.clone()]] = shape;
                    ref_strides[comptime![reverse.clone()]] = stride_curr;

                    stride_curr *= ref_shape[comptime![reverse]];
                }

                LocalArgs::new(
                    ref_shape.to_slice(),
                    ref_strides.to_slice(),
                    layout.tensor.line_size(),
                )
            }
            VirtualLayout::Reshaped(start) => {
                let mut stride_curr = 1u32;

                #[unroll]
                #[allow(clippy::clone_on_copy)]
                for i in 0..config.rank {
                    let reverse = comptime![reverse_index(config.rank, comptime![i.clone()])];
                    let reverse_u32_comptime = comptime!(unwrap_const_u32(reverse));
                    let arg = comptime![Arg::ScalarShape(start + reverse_u32_comptime)];
                    let shape = read_scalar_shape(inputs, comptime![arg.clone()]);

                    ref_shape[comptime![reverse_u32_comptime]] = shape;
                    ref_strides[comptime![reverse_u32_comptime]] = stride_curr;

                    stride_curr *= ref_shape[comptime![reverse_u32_comptime]];
                }

                LocalArgs::new(ref_shape.to_slice(), ref_strides.to_slice(), 1u32)
            }
        },
    }
}

fn unwrap_const_u32(elem: ExpandElementTyped<u32>) -> u32 {
    elem.constant().map(|cons| cons.as_u32()).unwrap()
}

#[cube]
fn fuse(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    pos: u32,
    #[comptime] config: &FuseBlockConfig,
) {
    #[unroll]
    for index in 0..config.ops.len() {
        let op = comptime! { config.ops.index(index).clone() };
        set_polyfill::<NumericExpand<DYN_ELEM_ID>>(comptime![op.cmp_elem()]);

        match op {
            FuseOp::Add(op) => {
                add::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Div(op) => {
                div::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Sub(op) => {
                sub::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Mul(op) => {
                mul::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Powf(op) => {
                powf::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Erf(op) => {
                erf::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Abs(op) => {
                abs::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Log(op) => {
                log::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Log1p(op) => {
                log1p::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Recip(op) => {
                recip::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Assign(op) => {
                assign::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Exp(op) => {
                exp::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Cos(op) => {
                cos::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Sin(op) => {
                sin::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Tanh(op) => {
                tanh::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Equal(op) => {
                equal::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Greater(op) => {
                greater::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::GreaterEqual(op) => greater_equal::<NumericExpand<DYN_ELEM_ID>>(
                inputs, outputs, locals, pos, op, config,
            ),
            FuseOp::Lower(op) => {
                lower::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::LowerEqual(op) => {
                lower_equal::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::ConditionalAssign {
                cond,
                lhs,
                rhs,
                out,
            } => conditional_assign::<NumericExpand<DYN_ELEM_ID>>(
                inputs, outputs, locals, pos, cond, lhs, rhs, out, config,
            ),
            FuseOp::Gather {
                input,
                indices,
                output,
                dim,
            } => gather::<NumericExpand<DYN_ELEM_ID>>(
                inputs, outputs, locals, pos, dim, input, indices, output, config,
            ),
            FuseOp::Select {
                input,
                indices,
                output,
                dim,
            } => select_indices::<NumericExpand<DYN_ELEM_ID>>(
                inputs, outputs, locals, pos, dim, input, indices, output, config,
            ),
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
            #[comptime] op: BinaryFuseArgs,
            #[comptime] config: &FuseBlockConfig,
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
            #[comptime] op: BinaryFuseArgs,
            #[comptime] config: &FuseBlockConfig,
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
            #[comptime] op: BinaryFuseArgs,
            #[comptime] config: &FuseBlockConfig,
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
            #[comptime] op: UnaryFuseArgs,
            #[comptime] config: &FuseBlockConfig,
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
    #[comptime] op: UnaryFuseArgs,
    #[comptime] config: &FuseBlockConfig,
) {
    let input = read::<C>(inputs, outputs, locals, write_pos, op.input, config);

    write::<C>(inputs, outputs, locals, write_pos, input, op.out, config);
}

#[cube]
fn gather<C: Numeric>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    write_pos: u32,
    #[comptime] dim: u32,
    #[comptime] input: Arg,
    #[comptime] indices: Arg,
    #[comptime] output: Arg,
    #[comptime] config: &FuseBlockConfig,
) {
    let mut index = read::<u32>(inputs, outputs, locals, write_pos, indices, config);
    let (pos, _precision) = comptime! {
        match input {
            Arg::Input(pos, precision, _) => (pos, precision),
            _ => panic!("Input tensor isn't an input"),
        }
    };
    let line_size = locals.ref_line_size;
    let stride = global_stride(inputs, dim, pos);

    index *= Line::new(stride);

    if comptime![dim > 0] {
        let index_before = global_offset(
            inputs,
            outputs,
            locals,
            write_pos,
            comptime!(input.clone()),
            comptime![Some((0u32, dim))],
            config,
        );
        index += Line::new(index_before);
    }

    if comptime![dim + 1 < config.rank] {
        let index_after = global_offset(
            inputs,
            outputs,
            locals,
            write_pos,
            input,
            comptime![Some((dim + 1, config.rank))],
            config,
        );
        index += Line::new(index_after);
    }

    let mut result = Line::empty(line_size);

    #[unroll]
    for i in 0..line_size {
        let index = index[i];

        let input = read_input::<C>(inputs, locals, pos, index, LayoutInfo::IsRef, config, None);
        result[i] = input[0];
    }

    write::<C>(inputs, outputs, locals, write_pos, result, output, config);
}

#[cube]
fn select_indices<C: Numeric>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    write_pos: u32,
    #[comptime] dim: u32,
    #[comptime] input: Arg,
    #[comptime] indices: Arg,
    #[comptime] output: Arg,
    #[comptime] config: &FuseBlockConfig,
) {
    let (line_size_ref, stride_dim_ref, shape_dim_ref) = (
        locals.ref_line_size,
        locals.ref_strides[dim],
        locals.ref_shape[dim],
    );

    let pos_input = comptime! {
        match input {
            Arg::Input(pos, ..) => pos,
            _ => panic!("Input tensor isn't an input"),
        }
    };
    let pos_indices = match indices {
        Arg::Input(pos, ..) => pos,
        _ => panic!("Indices tensor isn't an input"),
    };

    let stride_input_dim = global_stride(inputs, dim, pos_input);

    let mut index = 0u32;
    let mut result = Line::empty(line_size_ref);

    if comptime![dim != config.rank - 1] {
        // In this scenario the select is actually broadcasted along the axis we're working on.
        //
        // Therefore the same indices are used to fetch multiple entries in the input tensor.

        if comptime![dim > 0] {
            let index_before = global_offset(
                inputs,
                outputs,
                locals,
                write_pos,
                comptime!(input.clone()),
                comptime![Some((0u32, dim))],
                config,
            );
            index += index_before;
        }

        if comptime![dim + 1 < config.rank] {
            let index_after = global_offset(
                inputs,
                outputs,
                locals,
                write_pos,
                comptime!(input.clone()),
                comptime![Some((dim + 1, config.rank))],
                config,
            );
            index += index_after;
        }

        let stride_input_line = global_stride(inputs, comptime![config.rank - 1], pos_input);
        let write_pos_input = write_pos * line_size_ref;
        let coordinate_dim = write_pos_input / stride_dim_ref % shape_dim_ref;
        let offset_dim = read_input::<u32>(
            inputs,
            locals,
            pos_indices,
            coordinate_dim,
            LayoutInfo::IsRef,
            config,
            None,
        );

        index += offset_dim[0] * stride_input_dim;

        #[unroll]
        for i in 0..line_size_ref {
            let input = read_input::<C>(
                inputs,
                locals,
                pos_input,
                index + i * stride_input_line,
                LayoutInfo::IsRef,
                config,
                None,
            );
            result[i] = input[0];
        }
    } else {
        // In this scenario the select is actually performed on the last dimension we're working on.
        //
        // Therefore we need to fetch multiple indices that correspond to different entries in the
        // input tensor.

        if comptime![dim > 0] {
            let index_before = global_offset(
                inputs,
                outputs,
                locals,
                write_pos,
                comptime!(input.clone()),
                comptime![Some((0u32, dim))],
                config,
            );
            index += index_before;
        }

        if comptime![dim + 1 < config.rank] {
            let index_after = global_offset(
                inputs,
                outputs,
                locals,
                write_pos,
                input,
                comptime![Some((dim + 1, config.rank))],
                config,
            );
            index += index_after;
        }

        let write_pos_indices = write_pos * line_size_ref;

        #[unroll]
        for i in 0..line_size_ref {
            let coordinate_dim = (write_pos_indices + i) / stride_dim_ref % shape_dim_ref;
            let offset_dim = read_input::<u32>(
                inputs,
                locals,
                pos_indices,
                coordinate_dim,
                LayoutInfo::IsRef,
                config,
                None,
            );

            let input = read_input::<C>(
                inputs,
                locals,
                pos_input,
                index + (offset_dim[0] * stride_input_dim),
                LayoutInfo::IsRef,
                config,
                None,
            );
            result[i] = input[0];
        }
    }

    write::<C>(inputs, outputs, locals, write_pos, result, output, config);
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
    #[comptime] config: &FuseBlockConfig,
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
