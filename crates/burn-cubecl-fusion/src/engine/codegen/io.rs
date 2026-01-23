//! This module declares input-output primitives to read and write values during kernel expansion.
use super::{DYN_ELEM_ID, ir::*, tensor::GlobalTensor};
use burn_std::quantization::QuantScheme;
use cubecl::quant::scheme::QuantLevel;
use cubecl::{
    intrinsic,
    ir::{ExpandElement, Variable},
    prelude::*,
    std::{FastDivmod, tensor::View},
};
use cubek::quantization::layout::{BlockScaledLayout, PerTensorLayout, ScalesLayout};
use serde::{Deserialize, Serialize};

/// Define how a tensor might be transformed at runtime.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum Transform {
    /// A reshape operation has been registered on a tensor.
    ///
    /// This enum entry contains a sequence of [arguments](FuseArg) that points to global scalars representing the
    /// new shape for the current tensor.
    Reshape(Vec<FuseArg>),
    /// Two axes have been swapped on a tensor.
    ///
    /// The enum entry contains those two axes.
    SwapDims(usize, usize),
}

/// Reads the value from the [arg](FuseArg) and cast it to the generic cube primitive.
///
/// # Notes
///
/// The [global arguments](GlobalArgs) for both inputs and outputs as well as the
/// [local arguments](LocalArgs) need to be passed to this function.
///
/// This is because the [argument](FuseArg) might point to a global input, output or local variable
/// created during kernel expansion.
#[cube]
pub fn read<C: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    locals: &LocalArgs,
    ref_pos: usize,
    #[comptime] arg: FuseArg,
    #[comptime] config: &FuseBlockConfig,
) -> Line<C> {
    match arg {
        FuseArg::Input(pos, _precision, layout) => {
            let global = inputs.tensors.index(pos);
            let line_size = global.tensor.line_size();

            if comptime![!global.broadcasted && line_size != config.width] {
                read_input_aligned(inputs, locals, pos, ref_pos, layout, config, None)
            } else {
                read_input(inputs, locals, pos, ref_pos, layout, config, None)
            }
        }
        FuseArg::GlobalRegister(key, _precision) => Line::cast_from(outputs.registers.read(key)),
        FuseArg::Output(pos, _precision, layout) => {
            read_output(inputs, outputs, locals, pos, ref_pos, layout, config)
        }
        FuseArg::Local(pos, precision) => match comptime![precision] {
            FuseType::F64 => Line::cast_from(locals.l_f64.find(pos)),
            FuseType::F32 | FuseType::Flex32 => Line::cast_from(locals.l_f32.find(pos)),
            FuseType::F16 => Line::cast_from(locals.l_f16.find(pos)),
            FuseType::BF16 => Line::cast_from(locals.l_bf16.find(pos)),
            FuseType::U64 => Line::cast_from(locals.l_u64.find(pos)),
            FuseType::U32 => Line::cast_from(locals.l_u32.find(pos)),
            FuseType::U16 => Line::cast_from(locals.l_u16.find(pos)),
            FuseType::U8 => Line::cast_from(locals.l_u8.find(pos)),
            FuseType::I64 => Line::cast_from(locals.l_i64.find(pos)),
            FuseType::I32 => Line::cast_from(locals.l_i32.find(pos)),
            FuseType::I16 => Line::cast_from(locals.l_i16.find(pos)),
            FuseType::I8 => Line::cast_from(locals.l_i8.find(pos)),
            FuseType::Bool => Line::cast_from(locals.l_bool.find(pos)),
        },
        FuseArg::Scalar(..) => {
            let scalar = read_scalar::<C>(inputs, arg);
            Line::new(scalar)
        }
        FuseArg::ScalarShape(_) => {
            let scalar = read_scalar_shape(inputs, arg);
            Line::cast_from(scalar)
        }
        FuseArg::Literal(val, _precision) => Line::new(from_const_int::<C>(val)),
        FuseArg::InputReshaped {
            original,
            shape,
            broadcasted,
        } => match comptime![original.as_ref().clone()] {
            FuseArg::Input(pos, _precision, layout) => {
                let global = inputs.tensors.index(pos);
                let line_size = global.tensor.line_size();

                if comptime![!broadcasted && line_size != config.width] {
                    read_input_aligned(
                        inputs,
                        locals,
                        pos,
                        ref_pos,
                        layout,
                        config,
                        comptime![Some(Transform::Reshape(shape))],
                    )
                } else {
                    read_input(
                        inputs,
                        locals,
                        pos,
                        ref_pos,
                        layout,
                        config,
                        comptime![Some(Transform::Reshape(shape))],
                    )
                }
            }
            _ => comptime![panic!("Only input can be reshaped")],
        },
        FuseArg::InputSwapDims {
            original,
            dims,
            broadcasted,
        } => match comptime![original.as_ref().clone()] {
            FuseArg::Input(pos, _precision, layout) => {
                let global = inputs.tensors.index(pos);
                let line_size = global.tensor.line_size();

                if comptime![!broadcasted && line_size != config.width] {
                    read_input_aligned(
                        inputs,
                        locals,
                        pos,
                        ref_pos,
                        layout,
                        config,
                        comptime![Some(Transform::SwapDims(dims.0, dims.1))],
                    )
                } else {
                    read_input(
                        inputs,
                        locals,
                        pos,
                        ref_pos,
                        layout,
                        config,
                        comptime![Some(Transform::SwapDims(dims.0, dims.1))],
                    )
                }
            }
            _ => comptime![panic!("Only input can be swapped dims")],
        },
    }
}

/// Computes the offset for the current global tensor with a quantized layout.
///
/// The offset can be used to fetch the correct data from the quantized tensor as if it was in a
/// linear contiguous format.
#[cube]
fn index_offset_with_quant_layout(
    tensor: &GlobalTensor,
    locals: &LocalArgs,
    index: usize,
    #[comptime] rank: usize,
    #[comptime] scheme: QuantScheme,
) -> usize {
    let (start, end) = (0, rank - 1);
    let num_quants = scheme.num_quants();

    let offset_ref = index * locals.ref_line_size;
    let mut offset = 0;

    #[unroll]
    for i in start..end {
        let ogwl = offset_ref / locals.ref_strides[i];
        offset += ogwl % tensor.tensor.shape(i) * tensor.tensor.stride(i);
    }

    // Handle packed representation in last dim
    let ogwl = offset_ref / locals.ref_strides[end];
    let shape_last = tensor.tensor.shape(end).div_ceil(num_quants);
    let stride_last = tensor.tensor.stride(end);
    offset += (ogwl.div_ceil(num_quants)) % shape_last * stride_last;

    offset / tensor.tensor.line_size()
}

/// Reads a global quantized tensor at the given position.
///
/// # Notes
///
/// The values returned in the [Line] are not dequantized.
#[cube]
pub fn read_quantized<C: CubePrimitive>(
    inputs: &GlobalArgs,
    locals: &LocalArgs,
    ref_pos: usize,
    #[comptime] arg: FuseArg,
    #[comptime] config: &FuseBlockConfig,
    #[comptime] scheme: QuantScheme,
) -> Line<C> {
    match arg {
        FuseArg::Input(pos, _precision, _layout) => {
            let global = inputs.tensors.index(pos);

            let offset =
                index_offset_with_quant_layout(global, locals, ref_pos, config.rank, scheme);
            let val = global.tensor[offset];
            Line::cast_from(val)
        }
        _ => panic!("Not supported"),
    }
}

/// Reads a global scalar.
#[cube]
pub fn read_scalar<C: CubePrimitive>(inputs: &GlobalArgs, #[comptime] arg: FuseArg) -> C {
    match arg {
        FuseArg::Scalar(pos, _precision) => {
            let scalar = inputs.scalars.index(pos);
            scalar.get::<C>()
        }
        _ => comptime![panic!("Not a scalar")],
    }
}

/// Reads a global scalar that is used as a reshape position.
#[cube]
pub fn read_scalar_shape(inputs: &GlobalArgs, #[comptime] arg: FuseArg) -> usize {
    match arg {
        FuseArg::ScalarShape(pos) => inputs.reshapes[pos],
        _ => comptime![panic!("Not a scalar shape")],
    }
}

/// Reads an input tensor.
#[cube]
pub fn read_input<C: CubePrimitive>(
    inputs: &GlobalArgs,
    locals: &LocalArgs,
    #[comptime] pos: usize,
    ref_pos: usize,
    #[comptime] layout: LayoutInfo,
    #[comptime] config: &FuseBlockConfig,
    #[comptime] transform: Option<Transform>,
) -> Line<C> {
    let tensor = inputs.tensors.index(pos);
    let offset = match layout {
        LayoutInfo::SameAsRef => ref_pos,
        LayoutInfo::IsRef => ref_pos,
        LayoutInfo::Unknown => get_offset(inputs, locals, tensor, ref_pos, None, config, transform),
    };
    Line::cast_from(tensor.tensor[offset])
}

/// Returns a slice of data in the asked precision of the input tensor at the given position.
#[cube]
pub fn read_input_window<C: CubePrimitive>(
    inputs: &GlobalArgs,
    #[comptime] pos: usize,
    start: usize,
    end: usize,
) -> Slice<C> {
    let tensor = inputs.tensors.index(pos);
    let slice = tensor.tensor.slice(start, end);
    slice.downcast()
}

/// Returns the input as a slice.
#[cube]
pub fn input_as_slice<C: CubePrimitive>(inputs: &GlobalArgs, #[comptime] pos: usize) -> Slice<C> {
    let tensor = inputs.tensors.index(pos);
    let slice = tensor.tensor.to_slice();
    slice.downcast()
}

/// Returns the input tensor as a quantized scale view.
#[cube]
pub fn input_as_scales_view<C: CubePrimitive>(
    inputs: &GlobalArgs,
    #[comptime] pos: usize,
    #[comptime] tensor_pos: usize,
    #[comptime] level: QuantLevel,
    #[comptime] config: &FuseBlockConfig,
) -> View<C, usize> {
    set_polyfill_typed::<C, NumericExpand<DYN_ELEM_ID>>();
    let tensor = inputs.tensors.index(tensor_pos);
    let scales = inputs.tensors.index(pos);
    let tensor_len = tensor.tensor.len();
    let rank = config.rank;
    let layout = match level {
        QuantLevel::Tensor => ScalesLayout::new_PerTensor(PerTensorLayout::new(tensor_len)),
        QuantLevel::Block(block_size) => {
            let block_size = comptime![block_size.to_dim_vec(rank)];
            let mut tensor_shape = Sequence::new();
            let mut scales_strides = Sequence::new();
            #[unroll]
            for i in 0..rank {
                tensor_shape.push(FastDivmod::new_Fallback(tensor.tensor.shape(i)));
                scales_strides.push(scales.tensor.stride(i));
            }
            let line_size = scales.tensor.line_size();
            let layout = BlockScaledLayout::new(
                tensor_shape,
                tensor_len,
                scales_strides,
                block_size,
                line_size,
            );
            ScalesLayout::new_BlockScaled(layout)
        }
    };
    View::new::<Slice<C>, usize>(&scales.tensor.to_slice().downcast(), layout)
}

/// Reads the input tensor aligned.
#[cube]
pub fn read_input_aligned<C: CubePrimitive>(
    inputs: &GlobalArgs,
    locals: &LocalArgs,
    #[comptime] pos: usize,
    ref_pos: usize,
    #[comptime] layout: LayoutInfo,
    #[comptime] config: &FuseBlockConfig,
    #[comptime] transform: Option<Transform>,
) -> Line<C> {
    let mut result: Line<C> = Line::<C>::empty(config.width);
    let tensor = inputs.tensors.index(pos);

    match transform.clone() {
        Some(Transform::Reshape(shape)) => {
            // Very brute force, not really efficient, but not easy to optimize and not a very
            // frequent workflow.
            let ref_pos = ref_pos * config.width;
            #[unroll]
            for i in 0..config.width {
                let index = reshaped_index(
                    inputs,
                    locals,
                    ref_pos + i,
                    config.rank,
                    comptime![shape.clone()],
                );
                let index = reshaped_index_to_original_index(&tensor.tensor, index, config.rank);
                result[i] = C::cast_from(tensor.tensor[index][0])
            }
        }
        Some(Transform::SwapDims(dim1, dim2)) => {
            let offset =
                get_offset_aligned(inputs, locals, tensor, ref_pos, layout, config, transform);
            let i = comptime![swap_dims_transform(config.rank - 1, (dim1, dim2))];
            let stride = tensor.tensor.stride(i);

            #[unroll]
            for i in 0..config.width {
                let index = offset + i * stride;
                result[i] = C::cast_from(tensor.tensor[index][0])
            }
        }
        None => {
            let offset =
                get_offset_aligned(inputs, locals, tensor, ref_pos, layout, config, transform);
            let stride = tensor.tensor.stride(config.rank - 1);
            #[unroll]
            for i in 0..config.width {
                let index = offset + i * stride;
                result[i] = C::cast_from(tensor.tensor[index][0])
            }
        }
    }

    result
}

/// Computes the offset of the given [GlobalTensor] at on the reference position with a linear
/// layout.
#[cube]
pub fn get_offset_aligned(
    inputs: &GlobalArgs,
    locals: &LocalArgs,
    tensor: &GlobalTensor,
    ref_pos: usize,
    #[comptime] layout: LayoutInfo,
    #[comptime] config: &FuseBlockConfig,
    #[comptime] transform: Option<Transform>,
) -> usize {
    match layout {
        LayoutInfo::SameAsRef | LayoutInfo::IsRef => {
            (ref_pos * locals.ref_line_size) / tensor.tensor.line_size()
        }
        LayoutInfo::Unknown => get_offset(
            inputs,
            locals,
            tensor,
            ref_pos,
            None,
            config,
            comptime!(transform.clone()),
        ),
    }
}

/// Reads an output tensor.
#[cube]
pub fn read_output<C: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    locals: &LocalArgs,
    #[comptime] pos: usize,
    ref_pos: usize,
    #[comptime] layout: LayoutInfo,
    #[comptime] config: &FuseBlockConfig,
) -> Line<C> {
    let tensor = outputs.tensors.index(pos);
    let offset = match layout {
        LayoutInfo::SameAsRef => ref_pos,
        LayoutInfo::IsRef => ref_pos,
        LayoutInfo::Unknown => get_offset(inputs, locals, tensor, ref_pos, None, config, None),
    };
    Line::cast_from(tensor.tensor[offset])
}

#[cube]
/// Write the given value at the [arg](Arg) position.
pub fn write<C: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    locals: &mut LocalArgs,
    ref_pos: usize,
    value: Line<C>,
    #[comptime] arg: FuseArg,
    #[comptime] config: &FuseBlockConfig,
) {
    match arg {
        FuseArg::Output(pos, precision, layout) => {
            let tensor = outputs.tensors.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, locals, tensor, ref_pos, None, config, None)
                }
            };
            let tensor = outputs.tensors.index_mut(pos);
            set_polyfill::<NumericExpand<DYN_ELEM_ID>>(comptime![precision.into_type()]);

            tensor.tensor[offset] = Line::cast_from(value);
        }
        FuseArg::Local(..) => write_scalar::<C>(locals, value, arg),
        FuseArg::GlobalRegister(key, _precision) => {
            outputs.registers.write(key, Line::cast_from(value))
        }
        _ => comptime![panic!("Can't write into inputs and scalars")],
    }
}

#[cube]
/// Write the given value at the [arg](Arg) position.
pub fn write_scalar<C: CubePrimitive>(
    locals: &mut LocalArgs,
    value: Line<C>,
    #[comptime] arg: FuseArg,
) {
    match arg {
        FuseArg::Local(pos, precision) => match comptime![precision] {
            FuseType::F64 => locals.l_f64.insert(pos, Line::cast_from(value)),
            FuseType::F32 | FuseType::Flex32 => locals.l_f32.insert(pos, Line::cast_from(value)),
            FuseType::F16 => locals.l_f16.insert(pos, Line::cast_from(value)),
            FuseType::BF16 => locals.l_bf16.insert(pos, Line::cast_from(value)),
            FuseType::U64 => locals.l_u64.insert(pos, Line::cast_from(value)),
            FuseType::U32 => locals.l_u32.insert(pos, Line::cast_from(value)),
            FuseType::U16 => locals.l_u16.insert(pos, Line::cast_from(value)),
            FuseType::U8 => locals.l_u8.insert(pos, Line::cast_from(value)),
            FuseType::I64 => locals.l_i64.insert(pos, Line::cast_from(value)),
            FuseType::I32 => locals.l_i32.insert(pos, Line::cast_from(value)),
            FuseType::I16 => locals.l_i16.insert(pos, Line::cast_from(value)),
            FuseType::I8 => locals.l_i8.insert(pos, Line::cast_from(value)),
            FuseType::Bool => locals.l_bool.insert(pos, Line::cast_from(value)),
        },
        _ => comptime![panic!("Can't write into something else than scalars")],
    }
}

#[cube]
pub(crate) fn global_offset(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    locals: &LocalArgs,
    index: usize,
    #[comptime] arg: FuseArg,
    #[comptime] range: Option<(usize, usize)>,
    #[comptime] config: &FuseBlockConfig,
) -> usize {
    match arg {
        FuseArg::Input(pos, _precision, _layout) => {
            let tensor = inputs.tensors.index(pos);
            get_offset(inputs, locals, tensor, index, range, config, None)
        }
        FuseArg::Output(pos, _precision, _layout) => {
            let tensor = outputs.tensors.index(pos);
            get_offset(inputs, locals, tensor, index, range, config, None)
        }
        _ => panic!("Only input and output tensors have global offset."),
    }
}

#[cube]
fn get_offset(
    inputs: &GlobalArgs,
    locals: &LocalArgs,
    tensor: &GlobalTensor,
    ref_pos: usize,
    #[comptime] range: Option<(usize, usize)>,
    #[comptime] config: &FuseBlockConfig,
    #[comptime] transform: Option<Transform>,
) -> usize {
    index_offset_with_layout(
        inputs,
        tensor,
        locals,
        ref_pos,
        range,
        config.rank,
        transform,
    )
}

#[cube]
/// Gets the line size for a global tensor.
pub fn global_line_size(global: &GlobalArgs, #[comptime] pos: usize) -> comptime_type!(LineSize) {
    let tensor = global.tensors.index(pos);
    tensor.tensor.line_size()
}

#[cube]
/// Gets the rank for a global tensor.
pub fn global_rank(global: &GlobalArgs, #[comptime] pos: usize) -> usize {
    let tensor = global.tensors.index(pos);
    tensor.tensor.rank()
}

#[cube]
/// Gets the length for a global tensor.
pub fn global_len(global: &GlobalArgs, #[comptime] pos: usize) -> usize {
    let tensor = global.tensors.index(pos);
    tensor.tensor.len()
}

#[cube]
/// Gets the buffer length for a global tensor.
pub fn global_buffer_len(global: &GlobalArgs, #[comptime] pos: usize) -> usize {
    let tensor = global.tensors.index(pos);
    tensor.tensor.buffer_len()
}

#[cube]
/// Gets the reference tensor length.
pub fn ref_len(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    locals: &LocalArgs,
    #[comptime] config: &FuseBlockConfig,
) -> usize {
    match config.ref_layout.clone() {
        RefLayout::Concrete(arg) => match comptime![arg] {
            FuseArg::Input(index, _, _) => global_len(inputs, index),
            FuseArg::Output(index, _, _) => global_len(outputs, index),
            _ => panic!("Invalid concrete ref layout."),
        },
        RefLayout::Virtual(..) => num_elements(locals, config),
    }
}

#[cube]
/// Gets the reference buffer tensor length.
pub fn ref_buffer_len(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    locals: &LocalArgs,
    #[comptime] config: &FuseBlockConfig,
) -> usize {
    match config.ref_layout.clone() {
        RefLayout::Concrete(arg) => match comptime![arg] {
            FuseArg::Input(index, _, _) => global_buffer_len(inputs, index),
            FuseArg::Output(index, _, _) => global_buffer_len(outputs, index),
            _ => panic!("Invalid concrete ref layout."),
        },
        RefLayout::Virtual(VirtualLayout::SwapDims(arg, ..)) => match arg {
            FuseArg::Input(index, _, _) => global_buffer_len(inputs, index),
            FuseArg::Output(index, _, _) => global_buffer_len(outputs, index),
            _ => panic!("Invalid concrete ref layout."),
        },
        RefLayout::Virtual(VirtualLayout::Reshaped { .. }) => num_elements(locals, config),
        RefLayout::Virtual(VirtualLayout::Shape(..)) => num_elements(locals, config),
    }
}

#[cube]
/// Gets the reference number of elements.
pub fn num_elements(locals: &LocalArgs, #[comptime] config: &FuseBlockConfig) -> usize {
    let mut length = 1;

    for i in 0..config.rank {
        length *= locals.ref_shape[i];
    }

    length
}

#[cube]
/// Gets the reference axis shape.
pub fn ref_shape(locals: &LocalArgs, axis: usize) -> usize {
    locals.ref_shape[axis]
}

#[cube]
/// Gets the reference axis stride.
pub fn ref_stride(locals: &LocalArgs, axis: usize) -> usize {
    locals.ref_strides[axis]
}

#[cube]
/// Gets the reference line size.
pub fn ref_line_size(locals: &LocalArgs) -> comptime_type!(LineSize) {
    comptime![locals.ref_line_size]
}

#[cube]
/// Gets the given tensor axis shape.
pub fn global_shape(global: &GlobalArgs, axis: usize, #[comptime] pos: usize) -> usize {
    let tensor = global.tensors.index(pos);
    tensor.tensor.shape(axis)
}

#[cube]
/// Gets the given tensor axis stride.
pub fn global_stride(global: &GlobalArgs, dim: usize, #[comptime] pos: usize) -> usize {
    let tensor = global.tensors.index(pos);
    tensor.tensor.stride(dim)
}

#[cube]
fn index_offset_with_layout(
    inputs: &GlobalArgs,
    tensor: &GlobalTensor,
    locals: &LocalArgs,
    index: usize,
    #[comptime] range: Option<(usize, usize)>,
    #[comptime] rank: usize,
    #[comptime] transform: Option<Transform>,
) -> usize {
    match comptime![transform.clone()] {
        Some(Transform::Reshape(shape)) => {
            comptime![assert!(
                range.is_none(),
                "Can't get a range on a reshaped tensor."
            )];

            let index = index * locals.ref_line_size;
            let index = reshaped_index(inputs, locals, index, rank, shape);
            reshaped_index_to_original_index(&tensor.tensor, index, rank)
        }
        Some(Transform::SwapDims(dim1, dim2)) => {
            let (start, end) = comptime! {match range {
                Some(range) => range,
                None => (0, rank),
            }};

            let offset_ref = index * locals.ref_line_size;
            let mut offset = 0;

            #[unroll]
            for i in start..end {
                let index = comptime![swap_dims_transform(i, (dim1, dim2))];
                let ogwl = offset_ref / locals.ref_strides[i];
                offset += ogwl % tensor.tensor.shape(index) * tensor.tensor.stride(index);
            }

            offset / tensor.tensor.line_size()
        }
        None => {
            let (start, end) = comptime! {match range {
                Some(range) => range,
                None => (0, rank),
            }};

            let offset_ref = index * locals.ref_line_size;
            let mut offset = 0;

            #[unroll]
            for i in start..end {
                let ogwl = offset_ref / locals.ref_strides[i];
                offset += ogwl % tensor.tensor.shape(i) * tensor.tensor.stride(i);
            }

            offset / tensor.tensor.line_size()
        }
    }
}

pub(crate) fn swap_dims_transform(i: usize, dims: (usize, usize)) -> usize {
    if i == dims.0 {
        dims.1
    } else if i == dims.1 {
        dims.0
    } else {
        i
    }
}

#[cube]
#[allow(clippy::clone_on_copy)]
/// The index the input tensor would be at if it was contiguous.
fn reshaped_index(
    inputs: &GlobalArgs,
    locals: &LocalArgs,
    index: usize,
    #[comptime] rank: usize,
    #[comptime] shape: Vec<FuseArg>,
) -> usize {
    let mut offset = 0;
    let mut stride_curr = 1;

    #[unroll]
    for r in 0..rank {
        let i = reverse_index(rank, r).comptime();
        let arg = shape[i].clone();
        let shape_i = read_scalar_shape(inputs, arg);
        let ogwl = index / locals.ref_strides[i];

        offset += ogwl % shape_i * stride_curr;

        stride_curr *= shape_i;
    }

    offset
}

#[allow(unreachable_code)]
#[cube]
#[allow(clippy::clone_on_copy)]
fn reshaped_index_to_original_index<C: CubePrimitive>(
    original: &Tensor<Line<C>>,
    index_reshaped: usize,
    #[comptime] rank: usize,
) -> usize {
    let mut remaining = index_reshaped;
    let mut offset = 0;

    #[unroll]
    for r in 0..rank {
        let i = reverse_index(rank, r);
        let shape = original.shape(i);
        let stride = original.stride(i);

        let coordinate = remaining % shape;

        remaining /= shape;
        offset += coordinate * stride;
    }

    offset / original.line_size()
}

#[cube]
#[allow(unused_variables)]
pub(crate) fn reverse_index(
    #[comptime] rank: usize,
    #[comptime] iter: usize,
) -> comptime_type!(usize) {
    rank - iter - 1
}

/// Generic way to construct any [`CubePrimitive`] from an int. Used for fusion.
#[allow(unused_variables)]
#[cube]
fn from_const_int<C: CubePrimitive>(#[comptime] value: usize) -> C {
    intrinsic!(|scope| {
        ExpandElement::Plain(Variable::constant(value.into(), C::as_type(scope))).into()
    })
}

#[cube]
#[allow(clippy::extra_unused_type_parameters)]
fn set_polyfill_typed<C: CubePrimitive, Dyn: CubePrimitive>() {
    intrinsic!(|scope| {
        let elem_type = C::as_type(scope);
        set_polyfill::expand::<Dyn>(scope, elem_type);
    })
}
