use super::{io::*, ir::*};
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
pub fn fuse_on_write<E: Scalar, N: Size>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    write_pos: usize,
    write_values: Registry<FuseArg, Vector<E, N>>,
    #[comptime] write_args: Vec<FuseArg>,
    #[comptime] config: &FuseBlockConfig,
) {
    comment!("Fuse on write begin");
    // Write the values given as arguments.
    #[unroll]
    for i in 0..write_args.len() {
        let arg = comptime![write_args.get(i).unwrap().clone()];
        let val = write_values.find(arg.clone());

        write::<E, N>(inputs, outputs, locals, write_pos, val, arg, config);
    }

    fuse(inputs, outputs, locals, write_pos, config);
    comment!("Fuse on write end");
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
pub fn fuse_on_read<E: Scalar, N: Size>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    read_pos: usize,
    #[comptime] read_args: Sequence<FuseArg>,
    #[comptime] config: &FuseBlockConfig,
) -> Sequence<Vector<E, N>> {
    comment!("Fuse on read begin");
    fuse(inputs, outputs, locals, read_pos, config);

    let mut output = Sequence::new();

    #[unroll]
    for i in 0..read_args.len() {
        let arg = comptime![read_args.index(i).clone()];
        let value = read::<E, N>(inputs, outputs, locals, read_pos, arg, config);

        output.push(value);
    }

    comment!("Fuse on read end");
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
    comment!("Init locals begin");
    let mut ref_shape = Array::new(config.rank);
    let mut ref_strides = Array::new(config.rank);

    let locals = match config.ref_layout.clone() {
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
                    layout.tensor.vector_size(),
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
                    layout.tensor.vector_size(),
                )
            }
            _ => comptime![panic!("Invalid concrete ref layout.")],
        },
        RefLayout::Virtual(layout) => match layout {
            VirtualLayout::SwapDims(original, dims) => {
                let layout = match original.clone() {
                    FuseArg::Input(pos, ..) => inputs.tensors.index(pos),
                    FuseArg::Output(pos, ..) => outputs.tensors.index(pos),
                    _ => comptime![panic!("Unsupported")],
                };

                let mut stride_curr = 1;

                #[unroll]
                #[allow(clippy::clone_on_copy)]
                for i in 0..config.rank {
                    let reverse = reverse_index(config.rank, i);
                    let swap = comptime![swap_dims_transform(reverse, dims)];
                    let shape = layout.tensor.shape(swap.clone());

                    ref_shape[reverse] = shape;
                    ref_strides[reverse] = stride_curr;

                    stride_curr *= ref_shape[comptime![reverse]];
                }

                LocalArgs::new(
                    ref_shape.to_slice(),
                    ref_strides.to_slice(),
                    layout.tensor.vector_size(),
                )
            }
            VirtualLayout::Reshaped {
                reshape_pos,
                vector_size,
            } => {
                let mut stride_curr = 1;
                let start = reshape_pos * config.rank;

                #[unroll]
                #[allow(clippy::clone_on_copy)]
                for i in 0..config.rank {
                    let reverse = reverse_index(config.rank, i);
                    let arg = comptime![FuseArg::ScalarShape(start + reverse)];
                    let shape = read_scalar_shape(inputs, arg.clone());

                    ref_shape[comptime![reverse]] = shape;
                    ref_strides[comptime![reverse]] = stride_curr;

                    stride_curr *= ref_shape[comptime![reverse]];
                }

                LocalArgs::new(ref_shape.to_slice(), ref_strides.to_slice(), vector_size)
            }
            VirtualLayout::Runtime { pos } => {
                let start_shape = (pos * 2) * config.rank;
                let start_strides = start_shape + config.rank;

                #[unroll]
                for i in 0..config.rank {
                    let shape_index = start_shape + i;
                    let strides_index = start_strides + i;

                    ref_shape[i] = *inputs.runtime_layouts.index(shape_index);
                    ref_strides[i] = *inputs.runtime_layouts.index(strides_index);
                }

                LocalArgs::new(ref_shape.to_slice(), ref_strides.to_slice(), config.width)
            }
            VirtualLayout::Shape(original, vector_size) => {
                let layout = match original.clone() {
                    FuseArg::Input(pos, ..) => inputs.tensors.index(pos),
                    FuseArg::Output(pos, ..) => outputs.tensors.index(pos),
                    _ => comptime![panic!("Unsupported")],
                };
                let mut stride_curr = 1;

                #[unroll]
                #[allow(clippy::clone_on_copy)]
                for i in 0..config.rank {
                    let reverse = reverse_index(config.rank, i);
                    let shape = layout.tensor.shape(reverse);

                    ref_shape[comptime![reverse]] = shape;
                    ref_strides[comptime![reverse]] = stride_curr;

                    stride_curr *= ref_shape[comptime![reverse]];
                }

                LocalArgs::new(ref_shape.to_slice(), ref_strides.to_slice(), vector_size)
            }
        },
    };
    comment!("Init locals end");
    locals
}

#[cube]
/// Expands all [operations](FuseOp) registered in the [block config](FuseBlockConfig].
fn fuse(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    pos: usize,
    #[comptime] config: &FuseBlockConfig,
) {
    #[unroll]
    for index in 0..config.ops.len() {
        let op = config.ops[index].clone();
        let define!(E) = op.cmp_storage_ty();
        let size!(N) = config.width;

        match op {
            FuseOp::Add(op) => add::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Div(op) => div::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Sub(op) => sub::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Mul(op) => mul::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Powf(op) => powf::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Erf(op) => erf::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Sqrt(op) => sqrt::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Abs(op) => abs::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Log(op) => log::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Log1p(op) => log1p::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Recip(op) => recip::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Assign(op) => assign::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Exp(op) => exp::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Cos(op) => cos::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Sin(op) => sin::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Tanh(op) => tanh::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Equal(op) => equal::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Greater(op) => greater::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::GreaterEqual(op) => {
                greater_equal::<E, N>(inputs, outputs, locals, pos, op, config)
            }
            FuseOp::Lower(op) => lower::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::LowerEqual(op) => lower_equal::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::ConditionalAssign {
                cond,
                lhs,
                rhs,
                out,
            } => conditional_assign::<E, N>(
                inputs, outputs, locals, pos, cond, lhs, rhs, out, config,
            ),
            FuseOp::Gather {
                input,
                indices,
                output,
                dim,
            } => gather::<E, N>(
                inputs, outputs, locals, pos, dim, input, indices, output, config,
            ),
            FuseOp::Select {
                input,
                indices,
                output,
                dim,
            } => select_indices::<E, N>(
                inputs, outputs, locals, pos, dim, input, indices, output, config,
            ),
            FuseOp::Dequantize {
                values,
                params,
                output,
                scheme,
            } => dequantize::<E, N>(
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
            FuseOp::Rem(op) => rem::<E, N>(inputs, outputs, locals, pos, op, config),
            FuseOp::Clamp {
                input,
                min,
                max,
                out,
            } => clamp::<E, N>(inputs, outputs, locals, pos, input, min, max, out, config),
        }
    }
}

macro_rules! binary_op {
    ($ident:ident, $op:tt) => {
        #[cube]
        fn $ident<C: Numeric, N: Size>(
            inputs: &GlobalArgs,
            outputs: &mut GlobalArgs,
            locals: &mut LocalArgs,
            write_pos: usize,
            #[comptime] op: BinaryFuseArgs,
            #[comptime] config: &FuseBlockConfig,
        ) {
            let lhs = read::<C, N>(inputs, outputs, &locals, write_pos, op.lhs, config);
            let rhs = read::<C, N>(inputs, outputs, &locals, write_pos, op.rhs, config);
            let result = lhs $op rhs;

            write::<C, N>(inputs, outputs, locals, write_pos, result, op.out, config);
        }
    };
}

macro_rules! binary_func {
    ($ident:ident, $func:expr, $c:tt) => {
        #[cube]
        fn $ident<C: $c, N: Size>(
            inputs: &GlobalArgs,
            outputs: &mut GlobalArgs,
            locals: &mut LocalArgs,
            write_pos: usize,
            #[comptime] op: BinaryFuseArgs,
            #[comptime] config: &FuseBlockConfig,
        ) {
            let lhs = read::<C, N>(inputs, outputs, &locals, write_pos, op.lhs, config);
            let rhs = read::<C, N>(inputs, outputs, &locals, write_pos, op.rhs, config);
            let result = $func(lhs, rhs);

            write::<C, N>(inputs, outputs, locals, write_pos, result, op.out, config);
        }
    };
}

macro_rules! comparison_op {
    ($ident:ident, $op:tt) => {
        #[cube]
        fn $ident<C: Scalar + core::cmp::PartialOrd, N: Size>(
            inputs: &GlobalArgs,
            outputs: &mut GlobalArgs,
            locals: &mut LocalArgs,
            write_pos: usize,
            #[comptime] op: BinaryFuseArgs,
            #[comptime] config: &FuseBlockConfig,
        ) {
            let lhs = read::<C, N>(inputs, outputs, &locals, write_pos, op.lhs, config);
            let rhs = read::<C, N>(inputs, outputs, &locals, write_pos, op.rhs, config);
            let result = Vector::new(lhs $op rhs);

            write::<bool, N>(inputs, outputs, locals, write_pos, result, op.out, config);
        }
    };
}

macro_rules! unary_func {
    ($ident:ident, $func:expr, $c:tt) => {
        #[cube]
        fn $ident<C: $c, N: Size>(
            inputs: &GlobalArgs,
            outputs: &mut GlobalArgs,
            locals: &mut LocalArgs,
            write_pos: usize,
            #[comptime] op: UnaryFuseArgs,
            #[comptime] config: &FuseBlockConfig,
        ) {
            let input = read::<C, N>(inputs, outputs, &locals, write_pos, op.input, config);
            let result = $func(input);

            write::<C, N>(inputs, outputs, locals, write_pos, result, op.out, config);
        }
    };
}

#[cube]
fn assign<C: Scalar, N: Size>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    write_pos: usize,
    #[comptime] op: UnaryFuseArgs,
    #[comptime] config: &FuseBlockConfig,
) {
    let input = read::<C, N>(inputs, outputs, locals, write_pos, op.input, config);

    write::<C, N>(inputs, outputs, locals, write_pos, input, op.out, config);
}

#[cube]
fn gather<C: Numeric, N: Size>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    write_pos: usize,
    #[comptime] dim: usize,
    #[comptime] input: FuseArg,
    #[comptime] indices: FuseArg,
    #[comptime] output: FuseArg,
    #[comptime] config: &FuseBlockConfig,
) {
    let vector_size = locals.ref_vector_size;

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

    let mut index = 0;
    let mut result = Vector::<C, N>::empty();

    if comptime![dim > 0] {
        let index_before = global_offset(
            inputs,
            outputs,
            locals,
            write_pos,
            input.clone(),
            comptime![Some((0, dim))],
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
        comptime![Some((0, config.rank))],
        config,
    );

    // TODO: new IR to differentiate between Gather and GatherBroadcasted at comptime?
    let stride_indices_vector = global_stride(inputs, config.rank - 1, pos_indices);

    if comptime![dim == config.rank - 1] {
        // Per-element indexing (along the dimension)
        #[unroll]
        for i in 0..vector_size {
            let offset = read_input::<u32, Const<1>>(
                inputs,
                locals,
                pos_indices,
                index_offset + i * stride_indices_vector,
                LayoutInfo::IsRef,
                config,
                None,
            );

            let input = read_input::<C, Const<1>>(
                inputs,
                locals,
                pos_input,
                index + (offset[0] as usize * stride_input_dim),
                LayoutInfo::IsRef,
                config,
                None,
            );
            result[i] = input[0];
        }
    } else {
        let stride_input_vector = global_stride(inputs, config.rank - 1, pos_input);

        #[unroll]
        for i in 0..vector_size {
            let offset = read_input::<u32, Const<1>>(
                inputs,
                locals,
                pos_indices,
                index_offset + i * stride_indices_vector,
                LayoutInfo::IsRef,
                config,
                None,
            );

            let current_index =
                index + (offset[0] as usize * stride_input_dim) + (i * stride_input_vector);

            let input = read_input::<C, Const<1>>(
                inputs,
                locals,
                pos_input,
                current_index,
                LayoutInfo::IsRef,
                config,
                None,
            );

            result[i] = input[0];
        }
    }

    write::<C, N>(inputs, outputs, locals, write_pos, result, output, config);
}

#[cube]
fn select_indices<C: Numeric, N: Size>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    write_pos: usize,
    #[comptime] dim: usize,
    #[comptime] input: FuseArg,
    #[comptime] indices: FuseArg,
    #[comptime] output: FuseArg,
    #[comptime] config: &FuseBlockConfig,
) {
    let (vector_size_ref, stride_dim_ref, shape_dim_ref) = (
        locals.ref_vector_size,
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

    let mut index = 0;
    let mut result = Vector::empty();

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
                input.clone(),
                comptime![Some((0, dim))],
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
                input.clone(),
                comptime![Some((dim + 1, config.rank))],
                config,
            );
            index += index_after;
        }

        let stride_input_vector = global_stride(inputs, comptime![config.rank - 1], pos_input);
        let write_pos_input = write_pos * vector_size_ref;
        let coordinate_dim = write_pos_input / stride_dim_ref % shape_dim_ref;
        let offset_dim = read_input::<u32, Const<1>>(
            inputs,
            locals,
            pos_indices,
            coordinate_dim,
            LayoutInfo::IsRef,
            config,
            None,
        );

        index += offset_dim[0] as usize * stride_input_dim;

        #[unroll]
        for i in 0..vector_size_ref {
            let input = read_input::<C, Const<1>>(
                inputs,
                locals,
                pos_input,
                index + i * stride_input_vector,
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
                input.clone(),
                comptime![Some((0, dim))],
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

        let write_pos_indices = write_pos * vector_size_ref;

        #[unroll]
        for i in 0..vector_size_ref {
            let coordinate_dim = (write_pos_indices + i) / stride_dim_ref % shape_dim_ref;
            let offset_dim = read_input::<u32, Const<1>>(
                inputs,
                locals,
                pos_indices,
                coordinate_dim,
                LayoutInfo::IsRef,
                config,
                None,
            );

            let input = read_input::<C, Const<1>>(
                inputs,
                locals,
                pos_input,
                index + (offset_dim[0] as usize * stride_input_dim),
                LayoutInfo::IsRef,
                config,
                None,
            );
            result[i] = input[0];
        }
    }

    write::<C, N>(inputs, outputs, locals, write_pos, result, output, config);
}

#[cube]
fn conditional_assign<C: Scalar, N: Size>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    write_pos: usize,
    #[comptime] cond: FuseArg,
    #[comptime] lhs: FuseArg,
    #[comptime] rhs: FuseArg,
    #[comptime] out: FuseArg,
    #[comptime] config: &FuseBlockConfig,
) {
    let cond = read::<bool, N>(inputs, outputs, locals, write_pos, cond, config);
    let lhs = read::<C, N>(inputs, outputs, locals, write_pos, lhs, config);
    let rhs = read::<C, N>(inputs, outputs, locals, write_pos, rhs, config);
    let result = select_many(cond, lhs, rhs);

    write::<C, N>(inputs, outputs, locals, write_pos, result, out, config);
}

#[cube]
fn clamp<C: Numeric, N: Size>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    write_pos: usize,
    #[comptime] input: FuseArg,
    #[comptime] min: FuseArg,
    #[comptime] max: FuseArg,
    #[comptime] out: FuseArg,
    #[comptime] config: &FuseBlockConfig,
) {
    let input = read::<C, N>(inputs, outputs, locals, write_pos, input, config);
    let min = read::<C, N>(inputs, outputs, locals, write_pos, min, config);
    let max = read::<C, N>(inputs, outputs, locals, write_pos, max, config);
    let result = cubecl::prelude::clamp(input, min, max);

    write::<C, N>(inputs, outputs, locals, write_pos, result, out, config);
}

#[cube]
#[allow(clippy::explicit_counter_loop)]
fn dequantize<C: Float, N: Size>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    write_pos: usize,
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

    let quant_ty = comptime![match scheme.store {
        QuantStore::Native => match scheme.value {
            QuantValue::Q8F | QuantValue::Q8S => StorageType::Scalar(ElemType::UInt(UIntKind::U8)),
            QuantValue::E4M3 => StorageType::Scalar(ElemType::Float(FloatKind::E4M3)),
            QuantValue::E5M2 => StorageType::Scalar(ElemType::Float(FloatKind::E5M2)),
            QuantValue::Q4F
            | QuantValue::Q4S
            | QuantValue::Q2F
            | QuantValue::Q2S
            | QuantValue::E2M1 => unreachable!("Can't store native sub-byte values"),
        },
        QuantStore::PackedU32(_) => ElemType::UInt(UIntKind::U32).into(),
        QuantStore::PackedNative(_) => match scheme.value {
            QuantValue::E2M1 => StorageType::Packed(ElemType::Float(FloatKind::E4M3), 2),
            other => panic!("{other:?} doesn't support native packing"),
        },
    }];
    let param_ty = comptime![match scheme.param {
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
    }];
    let q_vector_size = N::value().comptime() / scheme.num_quants();

    let define!(QStoreType) = quant_ty;
    let size!(QStoreSize) = q_vector_size;
    let define!(QParamType) = param_ty;
    let size!(NumQuant) = scheme.num_quants();

    let tensor_pos = comptime!(match input {
        FuseArg::Input(pos, _, _) => pos,
        _ => panic!("Not supported"),
    });
    let pos = comptime!(match scales {
        FuseArg::Input(pos, ..) => pos,
        _ => unreachable!(""),
    });

    let num_quants = scheme.num_quants();

    let input =
        read_quantized::<QStoreType, QStoreSize>(inputs, locals, write_pos, input, config, scheme);

    let scales =
        input_as_scales_view::<QParamType, Const<1>>(inputs, pos, tensor_pos, scheme.level, config);

    let result = dequantize_symmetric_packed_value_at::<
        C,
        NumQuant,
        QParamType,
        QStoreType,
        QStoreSize,
    >(write_pos * num_quants, input, &scales, scheme);

    let mut vector = Vector::empty();

    #[unroll]
    for i in 0..q_vector_size {
        let value = result[i];

        #[unroll]
        for j in 0..num_quants {
            let index = i * num_quants + j;
            vector[index] = value[j];
        }
    }

    write::<C, N>(inputs, outputs, locals, write_pos, vector, output, config);
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

binary_func!(powf, Vector::<C, N>::powf, Float);
binary_func!(rem, Vector::<C, N>::rem, Float);

unary_func!(exp, Vector::<C, N>::exp, Float);
unary_func!(log, Vector::<C, N>::ln, Float);
unary_func!(log1p, Vector::<C, N>::log1p, Float);
unary_func!(sqrt, Vector::<C, N>::sqrt, Float);
unary_func!(cos, Vector::<C, N>::cos, Float);
unary_func!(sin, Vector::<C, N>::sin, Float);
unary_func!(tanh, Vector::<C, N>::tanh, Float);
unary_func!(erf, Vector::<C, N>::erf, Float);
unary_func!(recip, Vector::<C, N>::recip, Float);
unary_func!(abs, Vector::<C, N>::abs, Numeric);
