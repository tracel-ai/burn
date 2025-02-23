use super::{ir::*, tensor::GlobalTensor, DYN_ELEM_ID};
use cubecl::{
    ir::{ExpandElement, Variable},
    prelude::*,
    unexpanded,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum Transform {
    Reshape(Sequence<Arg>),
    SwapDim(u32, u32),
}

#[cube]
/// Read the value from the [arg](Arg) and cast it to the generic cube primitive.
pub fn read<C: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    locals: &LocalArgs,
    ref_pos: u32,
    #[comptime] arg: Arg,
    #[comptime] config: &ElemwiseConfig,
) -> Line<C> {
    match arg {
        Arg::Input(pos, _precision, layout) => {
            let global = inputs.tensors.index(pos);
            let line_size = global.tensor.line_size();

            if comptime![!global.broadcasted && line_size != config.width as u32] {
                read_input_aligned(inputs, outputs, pos, ref_pos, layout, config, None)
            } else {
                read_input(inputs, outputs, pos, ref_pos, layout, config, None)
            }
        }
        Arg::Output(pos, _precision, layout) => {
            read_output(inputs, outputs, pos, ref_pos, layout, config)
        }
        Arg::Local(pos, precision) => match comptime![precision] {
            ElemwisePrecision::F32 => Line::cast_from(locals.l_f32.find(pos)),
            ElemwisePrecision::F16 => Line::cast_from(locals.l_f16.find(pos)),
            ElemwisePrecision::BF16 => Line::cast_from(locals.l_bf16.find(pos)),
            ElemwisePrecision::U64 => Line::cast_from(locals.l_u64.find(pos)),
            ElemwisePrecision::U32 => Line::cast_from(locals.l_u32.find(pos)),
            ElemwisePrecision::U16 => Line::cast_from(locals.l_u16.find(pos)),
            ElemwisePrecision::U8 => Line::cast_from(locals.l_u8.find(pos)),
            ElemwisePrecision::I64 => Line::cast_from(locals.l_i64.find(pos)),
            ElemwisePrecision::I32 => Line::cast_from(locals.l_i32.find(pos)),
            ElemwisePrecision::I16 => Line::cast_from(locals.l_i16.find(pos)),
            ElemwisePrecision::I8 => Line::cast_from(locals.l_i8.find(pos)),
            ElemwisePrecision::Bool => Line::cast_from(locals.l_bool.find(pos)),
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
                        outputs,
                        pos,
                        ref_pos,
                        layout,
                        config,
                        comptime![Some(Transform::Reshape(shape))],
                    )
                } else {
                    read_input(
                        inputs,
                        outputs,
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
                        outputs,
                        pos,
                        ref_pos,
                        layout,
                        config,
                        comptime![Some(Transform::SwapDim(dims.0, dims.1))],
                    )
                } else {
                    read_input(
                        inputs,
                        outputs,
                        pos,
                        ref_pos,
                        layout,
                        config,
                        comptime![Some(Transform::SwapDim(dims.0, dims.1))],
                    )
                }
            }
            _ => comptime![panic!("Only input can be swapped dims")],
        },
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
        Arg::ScalarShape(pos) => {
            let offset = comptime![inputs.scalars.len() - pos - 1];
            let scalar = inputs.scalars.index(offset);

            scalar.as_u32()
        }
        _ => comptime![panic!("Not a scalar shape")],
    }
}

#[cube]
pub fn read_input<C: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    #[comptime] pos: u32,
    ref_pos: u32,
    #[comptime] layout: LayoutInfo,
    #[comptime] config: &ElemwiseConfig,
    #[comptime] transform: Option<Transform>,
) -> Line<C> {
    let tensor = inputs.tensors.index(pos);
    let offset = match layout {
        LayoutInfo::SameAsRef => ref_pos,
        LayoutInfo::IsRef => ref_pos,
        LayoutInfo::Unknown => {
            get_offset(inputs, outputs, tensor, ref_pos, None, config, transform)
        }
    };
    Line::cast_from(tensor.tensor[offset])
}

#[cube]
pub fn read_input_aligned<C: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    #[comptime] pos: u32,
    ref_pos: u32,
    #[comptime] layout: LayoutInfo,
    #[comptime] config: &ElemwiseConfig,
    #[comptime] transform: Option<Transform>,
) -> Line<C> {
    let mut result: Line<C> = Line::<C>::empty(comptime![config.width as u32]);
    let tensor = inputs.tensors.index(pos);

    match comptime![transform.clone()] {
        Some(Transform::Reshape(shape)) => {
            let tensor_layout = match comptime![config.ref_layout.clone()] {
                Arg::Input(index, ..) => {
                    let layout = inputs.tensors.index(index);
                    &layout.tensor
                }
                Arg::Output(index, ..) => {
                    let layout = outputs.tensors.index(index);
                    &layout.tensor
                }
                _ => comptime![panic!("Invalid ref layout.")],
            };

            // Very brute force, not really efficient, but not easy to optimize and not a very
            // frequent workflow.
            let ref_pos = ref_pos * comptime![config.width as u32];
            #[unroll]
            for i in 0u32..comptime!(config.width as u32) {
                let index = reshaped_index(
                    inputs,
                    tensor_layout,
                    ref_pos + i,
                    config.rank,
                    comptime![shape.clone()],
                );
                let index = reshaped_index_to_original_index(&tensor.tensor, index, config.rank);
                result[i] = C::cast_from(tensor.tensor[index][0])
            }
        }
        Some(Transform::SwapDim(dim1, dim2)) => {
            let offset =
                get_offset_aligned(inputs, outputs, tensor, ref_pos, layout, config, transform);
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
                get_offset_aligned(inputs, outputs, tensor, ref_pos, layout, config, transform);
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
    outputs: &GlobalArgs,
    tensor: &GlobalTensor,
    ref_pos: u32,
    #[comptime] layout: LayoutInfo,
    #[comptime] config: &ElemwiseConfig,
    #[comptime] transform: Option<Transform>,
) -> u32 {
    match layout {
        LayoutInfo::SameAsRef | LayoutInfo::IsRef => {
            let line_size = match comptime![config.ref_layout.clone()] {
                Arg::Input(index, _precision, _) => {
                    let layout = inputs.tensors.index(index);
                    layout.tensor.line_size()
                }
                Arg::Output(index, _precision, _) => {
                    let layout = outputs.tensors.index(index);
                    layout.tensor.line_size()
                }
                _ => comptime![panic!("Invalid ref layout.")],
            };
            (ref_pos * line_size) / tensor.tensor.line_size()
        }
        LayoutInfo::Unknown => get_offset(
            inputs,
            outputs,
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
    pos: u32,
    ref_pos: u32,
    #[comptime] layout: LayoutInfo,
    #[comptime] config: &ElemwiseConfig,
) -> Line<C> {
    let tensor = outputs.tensors.index(pos);
    let offset = match layout {
        LayoutInfo::SameAsRef => ref_pos,
        LayoutInfo::IsRef => ref_pos,
        LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, None, config, None),
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
    #[comptime] config: &ElemwiseConfig,
) {
    match arg {
        Arg::Output(pos, precision, layout) => {
            let tensor = outputs.tensors.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                }
            };
            let tensor = outputs.tensors.index_mut(pos);
            set_polyfill::<NumericExpand<DYN_ELEM_ID>>(comptime![precision.into_elem()]);
            tensor.tensor[offset] = Line::cast_from(value);
        }
        Arg::Local(pos, precision) => match comptime![precision] {
            ElemwisePrecision::F32 => locals.l_f32.insert(pos, Line::cast_from(value)),
            ElemwisePrecision::F16 => locals.l_f16.insert(pos, Line::cast_from(value)),
            ElemwisePrecision::BF16 => locals.l_bf16.insert(pos, Line::cast_from(value)),
            ElemwisePrecision::U64 => locals.l_u64.insert(pos, Line::cast_from(value)),
            ElemwisePrecision::U32 => locals.l_u32.insert(pos, Line::cast_from(value)),
            ElemwisePrecision::U16 => locals.l_u16.insert(pos, Line::cast_from(value)),
            ElemwisePrecision::U8 => locals.l_u8.insert(pos, Line::cast_from(value)),
            ElemwisePrecision::I64 => locals.l_i64.insert(pos, Line::cast_from(value)),
            ElemwisePrecision::I32 => locals.l_i32.insert(pos, Line::cast_from(value)),
            ElemwisePrecision::I16 => locals.l_i16.insert(pos, Line::cast_from(value)),
            ElemwisePrecision::I8 => locals.l_i8.insert(pos, Line::cast_from(value)),
            ElemwisePrecision::Bool => locals.l_bool.insert(pos, Line::cast_from(value)),
        },
        _ => comptime![panic!("Can't write into inputs and scalars")],
    }
}

#[cube]
pub(crate) fn global_offset(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    index: u32,
    #[comptime] arg: Arg,
    #[comptime] range: Option<(u32, u32)>,
    #[comptime] config: &ElemwiseConfig,
) -> u32 {
    match arg {
        Arg::Input(pos, _precision, _layout) => {
            let tensor = inputs.tensors.index(pos);
            get_offset(inputs, outputs, tensor, index, range, config, None)
        }
        Arg::Output(pos, _precision, _layout) => {
            let tensor = outputs.tensors.index(pos);
            get_offset(inputs, outputs, tensor, index, range, config, None)
        }
        _ => todo!(),
    }
}

#[cube]
fn get_offset(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    tensor: &GlobalTensor,
    ref_pos: u32,
    #[comptime] range: Option<(u32, u32)>,
    #[comptime] config: &ElemwiseConfig,
    #[comptime] transform: Option<Transform>,
) -> u32 {
    match comptime![config.ref_layout.clone()] {
        Arg::Input(index, _precision, _) => {
            let layout = inputs.tensors.index(index);
            index_offset_with_layout(
                inputs,
                tensor,
                layout,
                ref_pos,
                range,
                config.rank,
                transform,
            )
        }
        Arg::Output(index, _precision, _) => {
            let layout = outputs.tensors.index(index);
            index_offset_with_layout(
                inputs,
                tensor,
                layout,
                ref_pos,
                range,
                config.rank,
                transform,
            )
        }
        _ => comptime![panic!("Invalid ref layout.")],
    }
}

#[cube]
pub fn global_line_size(global: &GlobalArgs, #[comptime] pos: u32) -> u32 {
    let tensor = global.tensors.index(pos);
    u32::cast_from(tensor.tensor.line_size())
}

#[cube]
pub fn global_length(global: &GlobalArgs, #[comptime] pos: u32) -> u32 {
    let tensor = global.tensors.index(pos);
    u32::cast_from(tensor.tensor.len())
}

#[cube]
pub fn global_rank(global: &GlobalArgs, #[comptime] pos: u32) -> u32 {
    let tensor = global.tensors.index(pos);
    tensor.tensor.rank()
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
    layout: &GlobalTensor,
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

            let index = index * layout.tensor.line_size();
            let index = reshaped_index(inputs, &layout.tensor, index, rank, shape);
            reshaped_index_to_original_index(&tensor.tensor, index, rank)
        }
        Some(Transform::SwapDim(dim1, dim2)) => {
            let (start, end) = comptime! {match range {
                Some(range) => range,
                None => (0u32, rank),
            }};

            let offset_ref = index * layout.tensor.line_size();
            let mut offset = 0u32;

            #[unroll]
            for i in start..end {
                let index = comptime![swap_dims_transform(&i, (dim1, dim2))];
                let ogwl = offset_ref / layout.tensor.stride(i);
                offset += ogwl % tensor.tensor.shape(index) * tensor.tensor.stride(index);
            }

            offset / tensor.tensor.line_size()
        }
        None => {
            let (start, end) = comptime! {match range {
                Some(range) => range,
                None => (0u32, rank),
            }};

            let offset_ref = index * layout.tensor.line_size();
            let mut offset = 0u32;

            for i in start..end {
                let ogwl = offset_ref / layout.tensor.stride(i);
                offset += ogwl % tensor.tensor.shape(i) * tensor.tensor.stride(i);
            }

            offset / tensor.tensor.line_size()
        }
    }
}

fn swap_dims_transform<I: Index + Clone>(i: &I, dims: (u32, u32)) -> u32 {
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
/// The index the input tensor would be at if it was contiguous.
fn reshaped_index(
    inputs: &GlobalArgs,
    layout: &Tensor<Line<NumericExpand<DYN_ELEM_ID>>>,
    index: u32,
    #[comptime] rank: u32,
    #[comptime] shape: Sequence<Arg>,
) -> u32 {
    let mut offset = 0u32;
    let mut stride_curr = 1u32;

    #[unroll]
    for r in 0..rank {
        let i = comptime![reverse_index(rank, r)];
        let arg = comptime![shape.index(i.clone())];
        let shape_i = read_scalar_shape(inputs, comptime![arg.clone()]);

        let ogwl = index / layout.stride(i);
        offset += ogwl % shape_i * stride_curr;

        stride_curr *= shape_i;
    }

    offset
}

#[cube]
fn reshaped_index_to_original_index<C: CubePrimitive>(
    original: &Tensor<Line<C>>,
    index_reshaped: u32,
    #[comptime] rank: u32,
) -> u32 {
    let mut remaining = index_reshaped;
    let mut offset = 0;

    #[unroll]
    for r in 0..rank {
        let i = comptime![reverse_index(rank, r)];
        let shape = original.shape(comptime![i.clone()]);
        let stride = original.stride(i);

        let coordinate = remaining % shape;

        remaining /= shape;
        offset += coordinate * stride;
    }

    offset / original.line_size()
}

fn reverse_index<Elem: Into<ExpandElementTyped<u32>>>(
    rank: u32,
    iter: Elem,
) -> ExpandElementTyped<u32> {
    let elem = iter.into();
    let elem = elem.constant().map(|cons| cons.as_u32()).unwrap();
    let result = rank - elem - 1;
    let scalar: Variable = result.into();
    let expand: ExpandElement = ExpandElement::Plain(scalar);

    expand.into()
}

/// Generic way to construct any [`CubePrimitive`] from an int. Used for fusion.
fn from_const_int<C: CubePrimitive>(_value: u32) -> C {
    unexpanded!()
}

mod from_const_int {
    use cubecl::ir::{ExpandElement, Scope, Variable};

    use cubecl::prelude::ExpandElementTyped;

    use super::CubePrimitive;

    pub fn expand<C: CubePrimitive>(scope: &mut Scope, value: u32) -> ExpandElementTyped<C> {
        let constant: ExpandElement = value.into();
        let constant_c = constant.as_const().unwrap().cast_to(C::as_elem(scope));
        ExpandElement::Plain(Variable::constant(constant_c)).into()
    }
}
