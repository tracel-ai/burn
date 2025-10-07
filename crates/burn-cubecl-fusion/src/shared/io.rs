use super::{DYN_ELEM_ID, ir::*, tensor::GlobalTensor};
use burn_tensor::quantization::QuantScheme;
use cubecl::{
    intrinsic,
    ir::{ExpandElement, Variable},
    prelude::*,
    std::{
        FastDivmod,
        tensor::{
            View,
            layout::{linear::LinearLayout, plain::PlainLayout},
        },
    },
};
use cubecl_quant::scheme::QuantLevel;
use cubecl_quant::{
    layout::{BlockScaledLayout, PerTensorLayout, ScalesLayout},
    scheme::QuantLevel,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum Transform {
    Reshape(Sequence<Arg>),
    SwapDims(u32, u32),
}

#[cube]
/// Read the value from the [arg](Arg) and cast it to the generic cube primitive.
pub fn read<C: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    locals: &LocalArgs,
    ref_pos: u32,
    #[comptime] arg: Arg,
    #[comptime] config: &FuseBlockConfig,
) -> Line<C> {
    match arg {
        Arg::Input(pos, _precision, layout) => {
            let global = inputs.tensors.index(pos);
            let line_size = global.tensor.line_size();

            if comptime![!global.broadcasted && line_size != config.width as u32] {
                read_input_aligned(inputs, locals, pos, ref_pos, layout, config, None)
            } else {
                read_input(inputs, locals, pos, ref_pos, layout, config, None)
            }
        }
        Arg::Output(pos, _precision, layout) => {
            read_output(inputs, outputs, locals, pos, ref_pos, layout, config)
        }
        Arg::Local(pos, precision) => match comptime![precision] {
            FusePrecision::F64 => Line::cast_from(locals.l_f64.find(pos)),
            FusePrecision::F32 | FusePrecision::Flex32 => Line::cast_from(locals.l_f32.find(pos)),
            FusePrecision::F16 => Line::cast_from(locals.l_f16.find(pos)),
            FusePrecision::BF16 => Line::cast_from(locals.l_bf16.find(pos)),
            FusePrecision::U64 => Line::cast_from(locals.l_u64.find(pos)),
            FusePrecision::U32 => Line::cast_from(locals.l_u32.find(pos)),
            FusePrecision::U16 => Line::cast_from(locals.l_u16.find(pos)),
            FusePrecision::U8 => Line::cast_from(locals.l_u8.find(pos)),
            FusePrecision::I64 => Line::cast_from(locals.l_i64.find(pos)),
            FusePrecision::I32 => Line::cast_from(locals.l_i32.find(pos)),
            FusePrecision::I16 => Line::cast_from(locals.l_i16.find(pos)),
            FusePrecision::I8 => Line::cast_from(locals.l_i8.find(pos)),
            FusePrecision::Bool => Line::cast_from(locals.l_bool.find(pos)),
        },
        Arg::Scalar(..) => {
            let scalar = read_scalar::<C>(inputs, arg);
            Line::new(scalar)
        }
        Arg::ScalarShape(_) => {
            let scalar = read_scalar_shape(inputs, arg);
            Line::cast_from(scalar)
        }
        Arg::Literal(val, _precision) => Line::new(from_const_int::<C>(val)),
        Arg::InputReshaped {
            original,
            shape,
            broadcasted,
        } => match comptime![original.as_ref().clone()] {
            Arg::Input(pos, _precision, layout) => {
                let global = inputs.tensors.index(pos);
                let line_size = global.tensor.line_size();

                if comptime![!broadcasted && line_size != config.width as u32] {
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
        Arg::InputSwapDims {
            original,
            dims,
            broadcasted,
        } => match comptime![original.as_ref().clone()] {
            Arg::Input(pos, _precision, layout) => {
                let global = inputs.tensors.index(pos);
                let line_size = global.tensor.line_size();

                if comptime![!broadcasted && line_size != config.width as u32] {
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

#[cube]
fn index_offset_with_quant_layout(
    tensor: &GlobalTensor,
    locals: &LocalArgs,
    index: u32,
    #[comptime] rank: u32,
    #[comptime] scheme: QuantScheme,
) -> u32 {
    let (start, end) = comptime![(0u32, rank - 1)];
    let num_quants = comptime!(scheme.num_quants() as u32);

    let offset_ref = index * locals.ref_line_size;
    let mut offset = 0u32;

    #[unroll]
    for i in start..end {
        let ogwl = offset_ref / locals.ref_strides[i];
        offset += ogwl % tensor.tensor.shape(i) * tensor.tensor.stride(i);
    }

    // Handle packed representation in last dim
    let ogwl = offset_ref / locals.ref_strides[end];
    let shape_last = tensor.tensor.shape(end) / num_quants;
    let stride_last = tensor.tensor.stride(end);
    offset += (ogwl / num_quants) % shape_last * stride_last;

    offset / tensor.tensor.line_size()
}

#[cube]
pub fn read_quantized<C: CubePrimitive>(
    inputs: &GlobalArgs,
    locals: &LocalArgs,
    ref_pos: u32,
    #[comptime] arg: Arg,
    #[comptime] config: &FuseBlockConfig,
    #[comptime] scheme: QuantScheme,
) -> Line<C> {
    match arg {
        Arg::Input(pos, _precision, _layout) => {
            let global = inputs.tensors.index(pos);

            let offset =
                index_offset_with_quant_layout(global, locals, ref_pos, config.rank, scheme);
            let val = global.tensor[offset];
            Line::cast_from(val)
        }
        _ => panic!("Not supported"),
    }
}

#[cube]
pub fn read_scalar<C: CubePrimitive>(inputs: &GlobalArgs, #[comptime] arg: Arg) -> C {
    match arg {
        Arg::Scalar(pos, _precision) => {
            let scalar = inputs.scalars.index(pos);
            scalar.read::<C>()
        }
        _ => comptime![panic!("Not a scalar")],
    }
}

#[cube]
pub fn read_scalar_shape(inputs: &GlobalArgs, #[comptime] arg: Arg) -> u32 {
    match arg {
        Arg::ScalarShape(pos) => *inputs.reshapes.index(pos),
        _ => comptime![panic!("Not a scalar shape")],
    }
}

#[cube]
pub fn read_input<C: CubePrimitive>(
    inputs: &GlobalArgs,
    locals: &LocalArgs,
    #[comptime] pos: u32,
    ref_pos: u32,
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

#[cube]
pub fn read_input_window<C: CubePrimitive>(
    inputs: &GlobalArgs,
    #[comptime] pos: u32,
    start: u32,
    end: u32,
) -> Slice<Line<C>> {
    let tensor = inputs.tensors.index(pos);
    let slice = tensor.tensor.slice(start, end);
    slice.try_cast_unchecked()
}

#[cube]
pub fn input_as_slice<C: CubePrimitive>(inputs: &GlobalArgs, #[comptime] pos: u32) -> Slice<C> {
    let tensor = inputs.tensors.index(pos);
    let slice = tensor.tensor.to_slice();
    slice.try_cast_unchecked()
}

#[cube]
pub fn input_as_linear_view<C: CubePrimitive>(
    inputs: &GlobalArgs,
    #[comptime] pos: u32,
) -> View<C, u32> {
    let slice = input_as_slice::<C>(inputs, pos);
    let layout = LinearLayout::new_Plain(PlainLayout::new(slice.len()));
    View::new::<Slice<C>, u32>(&slice, layout)
}

#[cube]
pub fn input_as_scales_view<C: CubePrimitive>(
    inputs: &GlobalArgs,
    #[comptime] pos: u32,
    #[comptime] tensor_pos: u32,
    #[comptime] level: QuantLevel,
    #[comptime] config: &FuseBlockConfig,
) -> View<C, u32> {
    let tensor = inputs.tensors.index(tensor_pos);
    let scales = inputs.tensors.index(pos);
    let tensor_len = tensor.tensor.len();
    let rank = config.rank;
    let layout = match level {
        QuantLevel::Tensor => ScalesLayout::new_PerTensor(PerTensorLayout::new(tensor_len)),
        QuantLevel::Block(block_size) => {
            let block_size = comptime![block_size.to_dim_vec(rank as usize)];
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
    View::new::<Slice<C>, u32>(&scales.tensor.to_slice().try_cast_unchecked(), layout)
}

#[cube]
pub fn read_input_aligned<C: CubePrimitive>(
    inputs: &GlobalArgs,
    locals: &LocalArgs,
    #[comptime] pos: u32,
    ref_pos: u32,
    #[comptime] layout: LayoutInfo,
    #[comptime] config: &FuseBlockConfig,
    #[comptime] transform: Option<Transform>,
) -> Line<C> {
    let mut result: Line<C> = Line::<C>::empty(comptime![config.width as u32]);
    let tensor = inputs.tensors.index(pos);

    match comptime![transform.clone()] {
        Some(Transform::Reshape(shape)) => {
            // Very brute force, not really efficient, but not easy to optimize and not a very
            // frequent workflow.
            let ref_pos = ref_pos * comptime![config.width as u32];
            #[unroll]
            for i in 0u32..comptime!(config.width as u32) {
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
            let i = comptime![swap_dims_transform(&(config.rank - 1), (dim1, dim2))];
            let stride = tensor.tensor.stride(comptime![i]);

            #[unroll]
            for i in 0u32..comptime!(config.width as u32) {
                let index = offset + i * stride;
                result[i] = C::cast_from(tensor.tensor[index][0])
            }
        }
        None => {
            let offset =
                get_offset_aligned(inputs, locals, tensor, ref_pos, layout, config, transform);
            let stride = tensor.tensor.stride(comptime![config.rank - 1]);
            #[unroll]
            for i in 0u32..comptime!(config.width as u32) {
                let index = offset + i * stride;
                result[i] = C::cast_from(tensor.tensor[index][0])
            }
        }
    }

    result
}

#[cube]
pub fn get_offset_aligned(
    inputs: &GlobalArgs,
    locals: &LocalArgs,
    tensor: &GlobalTensor,
    ref_pos: u32,
    #[comptime] layout: LayoutInfo,
    #[comptime] config: &FuseBlockConfig,
    #[comptime] transform: Option<Transform>,
) -> u32 {
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

#[cube]
pub fn read_output<C: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    locals: &LocalArgs,
    pos: u32,
    ref_pos: u32,
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
    ref_pos: u32,
    value: Line<C>,
    #[comptime] arg: Arg,
    #[comptime] config: &FuseBlockConfig,
) {
    match arg {
        Arg::Output(pos, precision, layout) => {
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
        Arg::Local(pos, precision) => match comptime![precision] {
            FusePrecision::F64 => locals.l_f64.insert(pos, Line::cast_from(value)),
            FusePrecision::F32 | FusePrecision::Flex32 => {
                locals.l_f32.insert(pos, Line::cast_from(value))
            }
            FusePrecision::F16 => locals.l_f16.insert(pos, Line::cast_from(value)),
            FusePrecision::BF16 => locals.l_bf16.insert(pos, Line::cast_from(value)),
            FusePrecision::U64 => locals.l_u64.insert(pos, Line::cast_from(value)),
            FusePrecision::U32 => locals.l_u32.insert(pos, Line::cast_from(value)),
            FusePrecision::U16 => locals.l_u16.insert(pos, Line::cast_from(value)),
            FusePrecision::U8 => locals.l_u8.insert(pos, Line::cast_from(value)),
            FusePrecision::I64 => locals.l_i64.insert(pos, Line::cast_from(value)),
            FusePrecision::I32 => locals.l_i32.insert(pos, Line::cast_from(value)),
            FusePrecision::I16 => locals.l_i16.insert(pos, Line::cast_from(value)),
            FusePrecision::I8 => locals.l_i8.insert(pos, Line::cast_from(value)),
            FusePrecision::Bool => locals.l_bool.insert(pos, Line::cast_from(value)),
        },
        _ => comptime![panic!("Can't write into inputs and scalars")],
    }
}

#[cube]
pub(crate) fn global_offset(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    locals: &LocalArgs,
    index: u32,
    #[comptime] arg: Arg,
    #[comptime] range: Option<(u32, u32)>,
    #[comptime] config: &FuseBlockConfig,
) -> u32 {
    match arg {
        Arg::Input(pos, _precision, _layout) => {
            let tensor = inputs.tensors.index(pos);
            get_offset(inputs, locals, tensor, index, range, config, None)
        }
        Arg::Output(pos, _precision, _layout) => {
            let tensor = outputs.tensors.index(pos);
            get_offset(inputs, locals, tensor, index, range, config, None)
        }
        _ => todo!(),
    }
}

#[cube]
fn get_offset(
    inputs: &GlobalArgs,
    locals: &LocalArgs,
    tensor: &GlobalTensor,
    ref_pos: u32,
    #[comptime] range: Option<(u32, u32)>,
    #[comptime] config: &FuseBlockConfig,
    #[comptime] transform: Option<Transform>,
) -> u32 {
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
pub fn global_line_size(global: &GlobalArgs, #[comptime] pos: u32) -> comptime_type!(u32) {
    let tensor = global.tensors.index(pos);
    tensor.tensor.line_size()
}

#[cube]
pub fn global_rank(global: &GlobalArgs, #[comptime] pos: u32) -> u32 {
    let tensor = global.tensors.index(pos);
    tensor.tensor.rank()
}

#[cube]
pub fn global_len(global: &GlobalArgs, #[comptime] pos: u32) -> u32 {
    let tensor = global.tensors.index(pos);
    tensor.tensor.len()
}

#[cube]
pub fn global_buffer_len(global: &GlobalArgs, #[comptime] pos: u32) -> u32 {
    let tensor = global.tensors.index(pos);
    tensor.tensor.buffer_len()
}

#[cube]
pub fn ref_len(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    locals: &LocalArgs,
    #[comptime] config: &FuseBlockConfig,
) -> u32 {
    match comptime![config.ref_layout.clone()] {
        RefLayout::Concrete(arg) => match comptime![arg] {
            Arg::Input(index, _, _) => global_len(inputs, index),
            Arg::Output(index, _, _) => global_len(outputs, index),
            _ => panic!("Invalid concrete ref layout."),
        },
        RefLayout::Virtual(..) => num_elements(locals, config),
    }
}

#[cube]
pub fn ref_buffer_len(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    locals: &LocalArgs,
    #[comptime] config: &FuseBlockConfig,
) -> u32 {
    match comptime![config.ref_layout.clone()] {
        RefLayout::Concrete(arg) => match comptime![arg] {
            Arg::Input(index, _, _) => global_buffer_len(inputs, index),
            Arg::Output(index, _, _) => global_buffer_len(outputs, index),
            _ => panic!("Invalid concrete ref layout."),
        },
        RefLayout::Virtual(VirtualLayout::SwapDims(arg, ..)) => match arg {
            Arg::Input(index, _, _) => global_buffer_len(inputs, index),
            Arg::Output(index, _, _) => global_buffer_len(outputs, index),
            _ => panic!("Invalid concrete ref layout."),
        },
        RefLayout::Virtual(VirtualLayout::Reshaped { .. }) => num_elements(locals, config),
        RefLayout::Virtual(VirtualLayout::Shape(..)) => num_elements(locals, config),
    }
}

#[cube]
pub fn num_elements(locals: &LocalArgs, #[comptime] config: &FuseBlockConfig) -> u32 {
    let mut length = 1u32;

    for i in 0..config.rank {
        length *= locals.ref_shape[i];
    }

    length
}

#[cube]
pub fn ref_shape(locals: &LocalArgs, dim: u32) -> u32 {
    locals.ref_shape[dim]
}

#[cube]
pub fn ref_stride(locals: &LocalArgs, dim: u32) -> u32 {
    locals.ref_strides[dim]
}

#[cube]
pub fn ref_line_size(locals: &LocalArgs) -> comptime_type!(u32) {
    comptime![locals.ref_line_size]
}

#[cube]
pub fn global_shape(global: &GlobalArgs, dim: u32, #[comptime] pos: u32) -> u32 {
    let tensor = global.tensors.index(pos);
    tensor.tensor.shape(dim)
}

#[cube]
pub fn global_stride(global: &GlobalArgs, dim: u32, #[comptime] pos: u32) -> u32 {
    let tensor = global.tensors.index(pos);
    tensor.tensor.stride(dim)
}

#[cube]
fn index_offset_with_layout(
    inputs: &GlobalArgs,
    tensor: &GlobalTensor,
    locals: &LocalArgs,
    index: u32,
    #[comptime] range: Option<(u32, u32)>,
    #[comptime] rank: u32,
    #[comptime] transform: Option<Transform>,
) -> u32 {
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
                None => (0u32, rank),
            }};

            let offset_ref = index * locals.ref_line_size;
            let mut offset = 0u32;

            #[unroll]
            for i in start..end {
                let index = comptime![swap_dims_transform(&i, (dim1, dim2))];
                let ogwl = offset_ref / locals.ref_strides[i];
                offset += ogwl % tensor.tensor.shape(index) * tensor.tensor.stride(index);
            }

            offset / tensor.tensor.line_size()
        }
        None => {
            let (start, end) = comptime! {match range {
                Some(range) => range,
                None => (0u32, rank),
            }};

            let offset_ref = index * locals.ref_line_size;
            let mut offset = 0u32;

            #[unroll]
            for i in start..end {
                let ogwl = offset_ref / locals.ref_strides[i];
                offset += ogwl % tensor.tensor.shape(i) * tensor.tensor.stride(i);
            }

            offset / tensor.tensor.line_size()
        }
    }
}

pub(crate) fn swap_dims_transform<I: Index + Clone>(i: &I, dims: (u32, u32)) -> u32 {
    let i_cloned: I = i.clone();
    let i = i_cloned.value().as_const().unwrap().as_u32();

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
    index: u32,
    #[comptime] rank: u32,
    #[comptime] shape: Sequence<Arg>,
) -> u32 {
    let mut offset = 0u32;
    let mut stride_curr = 1u32;

    #[unroll]
    for r in 0..rank {
        let i = reverse_index(rank, r);
        let arg = comptime![shape.index(i)];
        let shape_i = read_scalar_shape(inputs, comptime![arg.clone()]);
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
    index_reshaped: u32,
    #[comptime] rank: u32,
) -> u32 {
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
pub(crate) fn reverse_index(#[comptime] rank: u32, iter: u32) -> comptime_type!(u32) {
    intrinsic!(|_| {
        let elem = iter.constant().map(|cons| cons.as_u32()).unwrap();
        rank - elem - 1
    })
}

/// Generic way to construct any [`CubePrimitive`] from an int. Used for fusion.
#[allow(unused_variables)]
#[cube]
fn from_const_int<C: CubePrimitive>(#[comptime] value: u32) -> C {
    intrinsic!(|scope| {
        let constant: ExpandElement = value.into();
        let constant_c = constant.as_const().unwrap().cast_to(C::as_type(scope));
        ExpandElement::Plain(Variable::constant(constant_c)).into()
    })
}
