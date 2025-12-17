use super::{DYN_ELEM_ID, Q_PARAM_DYN_ELEM_ID, Q_STORE_DYN_ELEM_ID, io::*, ir::*};
use burn_std::quantization::{QuantScheme, QuantStore, QuantValue};
use cubecl::{
    ir::{ElemType, FloatKind, StorageType, UIntKind},
    prelude::*,
};
use cubek::quantization::{dequantize::dequantize_symmetric_packed_value_at, scheme::QuantMode};

#[cube]
/// Fuse element-wise operations at the given write position.
///
/// # Arguments
///
/// - `inputs`: Contains all readonly global kernel arguments.
/// - `outputs`: Contains all readwrite global kernel arguments.
/// - `locals`: Contains all local variables defined during kernel expansion.
/// - `write_pos`: The logical position the values are written to.
/// - `write_values`: The explicit values to write at the given position.
/// - `write_args`: The arguments associated to the `writes_values`.
/// - `config`: The current [fuse block configuration](FuseBlockConfig).
///
/// # Notes
///
/// The function will start by writing `write_values`.
pub fn fuse_on_write<E: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    write_pos: u32,
    write_values: Registry<FuseArg, Line<E>>,
    #[comptime] write_args: Sequence<FuseArg>,
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
///
/// # Arguments
///
/// - `inputs`: Contains all readonly global kernel arguments.
/// - `outputs`: Contains all readwrite global kernel arguments.
/// - `locals`: Contains all local variables defined during kernel expansion.
/// - `read_pos`: The logical position the values are read from.
/// - `read_args`: The arguments associated to the `read_pos`.
/// - `config`: The current [fuse block configuration](FuseBlockConfig).
///
/// # Returns
///
/// - A sequence of values associated to the given `read_args`.  
pub fn fuse_on_read<E: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    read_pos: u32,
    #[comptime] read_args: Sequence<FuseArg>,
    #[comptime] config: &FuseBlockConfig,
) -> Sequence<Line<E>> {
    fuse(inputs, outputs, locals, read_pos, config);

    let mut output = Sequence::new();

    #[unroll]
    for i in 0..read_args.len() {
        let arg = comptime![read_args.index(i).clone()];
        let value = read::<E>(inputs, outputs, locals, read_pos, arg, config);

        let value_line_size = value.line_size();
        let output_line_size = comptime!(config.width as u32);

        // We currently don't support broadcasting __across__ blocks.
        if comptime!(value_line_size != output_line_size) {
            let mut tmp = Line::<E>::empty(comptime!(config.width as u32));
            comptime!(
                assert_eq!(value_line_size, 1, "The input line_size must be 1 or the same as the config width.");
            );

            let val = value[0];

            #[unroll]
            for i in 0..comptime!(config.width as u32) {
                tmp[i] = val;
            }

            output.push(tmp);
        } else {
            output.push(value);
        }
    }

    output
}

#[cube]
/// Initializes [LocalArgs] given the input and output [arguments](GlobalArgs) with the [FuseBlockConfig].
///
/// # Notes
///
/// The goal is to resolve and cache the reference shape and strides, as it is used in many
/// different function during kernel expansion.
pub fn init_locals(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    #[comptime] config: &FuseBlockConfig,
) -> LocalArgs {
    let mut ref_shape = Array::new(config.rank);
    let mut ref_strides = Array::new(config.rank);

    match comptime![config.ref_layout.clone()] {
        RefLayout::Concrete(arg) => match comptime![arg] {
            FuseArg::Input(index, ..) => {
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
            FuseArg::Output(index, ..) => {
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
                    FuseArg::Input(pos, ..) => inputs.tensors.index(pos),
                    FuseArg::Output(pos, ..) => outputs.tensors.index(pos),
                    _ => comptime![panic!("Unsupported")],
                };

                let mut stride_curr = 1u32;

                #[unroll]
                #[allow(clippy::clone_on_copy)]
                for i in 0..config.rank {
                    let reverse = reverse_index(config.rank, i);
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
            VirtualLayout::Reshaped {
                reshape_pos,
                line_size,
            } => {
                let mut stride_curr = 1u32;
                let start = reshape_pos * config.rank;

                #[unroll]
                #[allow(clippy::clone_on_copy)]
                for i in 0..config.rank {
                    let reverse = reverse_index(config.rank, i);
                    let arg = comptime![FuseArg::ScalarShape(start + reverse)];
                    let shape = read_scalar_shape(inputs, comptime![arg.clone()]);

                    ref_shape[comptime![reverse]] = shape;
                    ref_strides[comptime![reverse]] = stride_curr;

                    stride_curr *= ref_shape[comptime![reverse]];
                }

                LocalArgs::new(ref_shape.to_slice(), ref_strides.to_slice(), line_size)
            }
            VirtualLayout::Shape(original, line_size) => {
                let layout = match comptime![original.clone()] {
                    FuseArg::Input(pos, ..) => inputs.tensors.index(pos),
                    FuseArg::Output(pos, ..) => outputs.tensors.index(pos),
                    _ => comptime![panic!("Unsupported")],
                };
                let mut stride_curr = 1u32;

                #[unroll]
                #[allow(clippy::clone_on_copy)]
                for i in 0..config.rank {
                    let reverse = reverse_index(config.rank, i);
                    let shape = layout.tensor.shape(reverse);

                    ref_shape[comptime![reverse]] = shape;
                    ref_strides[comptime![reverse]] = stride_curr;

                    stride_curr *= ref_shape[comptime![reverse]];
                }

                LocalArgs::new(ref_shape.to_slice(), ref_strides.to_slice(), line_size)
            }
        },
    }
}

#[cube]
/// Expands all [operations](FuseOp) registered in the [block config](FuseBlockConfig].
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
        set_polyfill::<NumericExpand<DYN_ELEM_ID>>(comptime![op.cmp_type()]);

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
            FuseOp::Sqrt(op) => {
                sqrt::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
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
            FuseOp::Dequantize {
                values,
                params,
                output,
                scheme,
            } => dequantize::<NumericExpand<DYN_ELEM_ID>>(
                inputs,
                outputs,
                locals,
                pos,
                values,
                params,
                output,
                scheme.scheme,
                config,
            ),
            FuseOp::Rem(op) => {
                rem::<NumericExpand<DYN_ELEM_ID>>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Clamp {
                input,
                min,
                max,
                out,
            } => clamp::<NumericExpand<DYN_ELEM_ID>>(
                inputs, outputs, locals, pos, input, min, max, out, config,
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
    #[comptime] input: FuseArg,
    #[comptime] indices: FuseArg,
    #[comptime] output: FuseArg,
    #[comptime] config: &FuseBlockConfig,
) {
    let line_size = locals.ref_line_size;

    let pos_input = comptime! {
        match input {
            FuseArg::Input(pos, ..) => pos,
            _ => panic!("Input tensor isn't an input"),
        }
    };
    let pos_indices = comptime! {
        match indices {
            FuseArg::Input(pos, ..) => pos,
            _ => panic!("Indices tensor isn't an input"),
        }
    };

    let stride_input_dim = global_stride(inputs, dim, pos_input);

    let mut index = 0u32;
    let mut result = Line::empty(line_size);

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

    let index_offset = global_offset(
        inputs,
        outputs,
        locals,
        write_pos,
        indices,
        comptime![Some((0u32, config.rank))],
        config,
    );

    if comptime![dim == config.rank - 1] {
        // Per-element indexing (along the dimension)
        #[unroll]
        for i in 0..line_size {
            let offset = read_input::<u32>(
                inputs,
                locals,
                pos_indices,
                index_offset + i,
                LayoutInfo::IsRef,
                config,
                None,
            );

            let input = read_input::<C>(
                inputs,
                locals,
                pos_input,
                index + (offset[0] * stride_input_dim),
                LayoutInfo::IsRef,
                config,
                None,
            );

            result[i] = input[0];
        }
    } else {
        // Shared index for whole line
        let stride_input_line = global_stride(inputs, comptime!(config.rank - 1), pos_input);

        let offset = read_input::<u32>(
            inputs,
            locals,
            pos_indices,
            index_offset,
            LayoutInfo::IsRef,
            config,
            None,
        );

        index += offset[0] * stride_input_dim;

        #[unroll]
        for i in 0..line_size {
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
    #[comptime] input: FuseArg,
    #[comptime] indices: FuseArg,
    #[comptime] output: FuseArg,
    #[comptime] config: &FuseBlockConfig,
) {
    let (line_size_ref, stride_dim_ref, shape_dim_ref) = (
        locals.ref_line_size,
        locals.ref_strides[dim],
        locals.ref_shape[dim],
    );

    let pos_input = comptime! {
        match input {
            FuseArg::Input(pos, ..) => pos,
            _ => panic!("Input tensor isn't an input"),
        }
    };
    let pos_indices = match indices {
        FuseArg::Input(pos, ..) => pos,
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
    #[comptime] cond: FuseArg,
    #[comptime] lhs: FuseArg,
    #[comptime] rhs: FuseArg,
    #[comptime] out: FuseArg,
    #[comptime] config: &FuseBlockConfig,
) {
    let cond = read::<bool>(inputs, outputs, locals, write_pos, cond, config);
    let lhs = read::<C>(inputs, outputs, locals, write_pos, lhs, config);
    let rhs = read::<C>(inputs, outputs, locals, write_pos, rhs, config);
    let result = select_many(cond, lhs, rhs);

    write::<C>(inputs, outputs, locals, write_pos, result, out, config);
}

#[cube]
fn clamp<C: Numeric>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    write_pos: u32,
    #[comptime] input: FuseArg,
    #[comptime] min: FuseArg,
    #[comptime] max: FuseArg,
    #[comptime] out: FuseArg,
    #[comptime] config: &FuseBlockConfig,
) {
    let input = read::<C>(inputs, outputs, locals, write_pos, input, config);
    let min = read::<C>(inputs, outputs, locals, write_pos, min, config);
    let max = read::<C>(inputs, outputs, locals, write_pos, max, config);
    let result = Line::<C>::clamp(input, min, max);

    write::<C>(inputs, outputs, locals, write_pos, result, out, config);
}

#[cube]
#[allow(clippy::explicit_counter_loop)]
fn dequantize<C: Float>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    write_pos: u32,
    #[comptime] input: FuseArg,
    #[comptime] scales: FuseArg,
    #[comptime] output: FuseArg,
    #[comptime] scheme: QuantScheme,
    #[comptime] config: &FuseBlockConfig,
) {
    comptime!(assert_eq!(
        scheme.mode,
        QuantMode::Symmetric,
        "Only symmetric quantization mode is supported."
    ));

    set_polyfill::<NumericExpand<Q_STORE_DYN_ELEM_ID>>(comptime![match scheme.store {
        QuantStore::Native => match scheme.value {
            QuantValue::Q8F | QuantValue::Q8S => StorageType::Scalar(ElemType::UInt(UIntKind::U8)),
            QuantValue::E4M3 => StorageType::Scalar(ElemType::Float(FloatKind::E4M3)),
            QuantValue::E5M2 => StorageType::Scalar(ElemType::Float(FloatKind::E5M2)),
            QuantValue::E2M1 => StorageType::Packed(ElemType::Float(FloatKind::E4M3), 2),
            QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S =>
                unreachable!("Can't store native sub-byte values"),
        },
        QuantStore::U32 => ElemType::UInt(UIntKind::U32).into(),
    }]);
    set_polyfill::<NumericExpand<Q_PARAM_DYN_ELEM_ID>>(comptime![match scheme.param {
        cubecl::quant::scheme::QuantParam::F32 =>
            StorageType::Scalar(ElemType::Float(FloatKind::F32)),
        cubecl::quant::scheme::QuantParam::F16 =>
            StorageType::Scalar(ElemType::Float(FloatKind::F16)),
        cubecl::quant::scheme::QuantParam::BF16 =>
            StorageType::Scalar(ElemType::Float(FloatKind::BF16)),
        cubecl::quant::scheme::QuantParam::UE8M0 =>
            StorageType::Scalar(ElemType::Float(FloatKind::UE8M0)),
        cubecl::quant::scheme::QuantParam::UE4M3 =>
            StorageType::Scalar(ElemType::Float(FloatKind::E4M3)),
    }]);

    let tensor_pos = comptime!(match input {
        FuseArg::Input(pos, _, _) => pos,
        _ => panic!("Not supported"),
    });
    let pos = comptime!(match scales {
        FuseArg::Input(pos, ..) => pos,
        _ => unreachable!(""),
    });
    let input = read_quantized::<NumericExpand<Q_STORE_DYN_ELEM_ID>>(
        inputs, locals, write_pos, input, config, scheme,
    );

    let line_size = input.line_size();
    let num_quants = comptime!(scheme.num_quants() as u32);

    let scales = input_as_scales_view::<NumericExpand<Q_PARAM_DYN_ELEM_ID>>(
        inputs,
        pos,
        tensor_pos,
        scheme.level,
        config,
    );
    let result = dequantize_symmetric_packed_value_at::<
        C,
        ElemExpand<Q_PARAM_DYN_ELEM_ID>,
        ElemExpand<Q_STORE_DYN_ELEM_ID>,
    >(write_pos * num_quants, input, &scales, scheme);

    let line_size_result = comptime!(num_quants * line_size);

    let line = if comptime!(line_size == 1) {
        result[0]
    } else {
        let mut line = Line::empty(line_size_result);

        // We have to do all index work as comptime because higher line sizes removes the
        // possibility to index dynamically on lines.
        let mut i = comptime!(0);

        #[unroll]
        for _ in 0..line_size {
            let mut j = comptime!(0);
            let value = result[i];

            #[unroll]
            for _ in 0..num_quants {
                let index = comptime!(i * num_quants + j);
                line[index] = value[j];
                comptime!(j += 1);
            }
            comptime!(i += 1);
        }

        line
    };

    write::<C>(inputs, outputs, locals, write_pos, line, output, config);
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
binary_func!(rem, Line::<C>::rem, Float);

unary_func!(exp, Line::<C>::exp, Float);
unary_func!(log, Line::<C>::log, Float);
unary_func!(log1p, Line::<C>::log1p, Float);
unary_func!(sqrt, Line::<C>::sqrt, Float);
unary_func!(cos, Line::<C>::cos, Float);
unary_func!(sin, Line::<C>::sin, Float);
unary_func!(tanh, Line::<C>::tanh, Float);
unary_func!(erf, Line::<C>::erf, Float);
unary_func!(recip, Line::<C>::recip, Float);
unary_func!(abs, Line::<C>::abs, Numeric);
