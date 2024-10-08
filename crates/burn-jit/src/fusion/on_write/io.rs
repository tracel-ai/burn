use super::ir::*;
use cubecl::{linalg::tensor::index_offset_with_layout, prelude::*};

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
        Arg::Input(pos, precision, layout) => match comptime![precision] {
            ElemwisePrecision::F32 => {
                let tensor = inputs.t_f32.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                Line::cast_from(tensor[offset])
            }
            ElemwisePrecision::F16 => {
                let tensor = inputs.t_f16.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                Line::cast_from(tensor[offset])
            }
            ElemwisePrecision::U32 => {
                let tensor = inputs.t_u32.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                Line::cast_from(tensor[offset])
            }
            ElemwisePrecision::I32 => {
                let tensor = inputs.t_i32.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                Line::cast_from(tensor[offset])
            }
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        Arg::Output(pos, precision, layout) => match comptime![precision] {
            ElemwisePrecision::F32 => {
                let tensor = outputs.t_f32.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                Line::cast_from(tensor[offset])
            }
            ElemwisePrecision::F16 => {
                let tensor = outputs.t_f16.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                Line::cast_from(tensor[offset])
            }
            ElemwisePrecision::U32 => {
                let tensor = outputs.t_u32.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                Line::cast_from(tensor[offset])
            }
            ElemwisePrecision::I32 => {
                let tensor = outputs.t_i32.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                Line::cast_from(tensor[offset])
            }
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        Arg::Local(pos, precision) => match comptime![precision] {
            ElemwisePrecision::F32 => Line::cast_from(locals.l_f32.find(pos)),
            ElemwisePrecision::F16 => Line::cast_from(locals.l_f16.find(pos)),
            ElemwisePrecision::U32 => Line::cast_from(locals.l_u32.find(pos)),
            ElemwisePrecision::I32 => Line::cast_from(locals.l_i32.find(pos)),
            ElemwisePrecision::Bool => Line::cast_from(locals.l_bool.find(pos)),
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        Arg::Scalar(pos, precision) => match comptime![precision] {
            ElemwisePrecision::F32 => Line::cast_from(*inputs.s_f32.index(pos)),
            ElemwisePrecision::F16 => Line::cast_from(*inputs.s_f16.index(pos)),
            ElemwisePrecision::U32 => Line::cast_from(*inputs.s_u32.index(pos)),
            ElemwisePrecision::I32 => Line::cast_from(*inputs.s_i32.index(pos)),
            ElemwisePrecision::BF16 => comptime![panic!("Can't write into inputs or scalars")],
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        Arg::Literal(val, _precision) => Line::cast_from(val.runtime()),
    }
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
        Arg::Output(pos, precision, layout) => match comptime![precision] {
            ElemwisePrecision::F32 => {
                let tensor = outputs.t_f32.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                let tensor = outputs.t_f32.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::F16 => {
                let tensor = outputs.t_f16.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                let tensor = outputs.t_f16.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::U32 => {
                let tensor = outputs.t_u32.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                let tensor = outputs.t_u32.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::I32 => {
                let tensor = outputs.t_i32.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                let tensor = outputs.t_i32.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        Arg::Local(pos, precision) => match comptime![precision] {
            ElemwisePrecision::F32 => locals.l_f32.insert(pos, Line::cast_from(value)),
            ElemwisePrecision::F16 => locals.l_f16.insert(pos, Line::cast_from(value)),
            ElemwisePrecision::U32 => locals.l_u32.insert(pos, Line::cast_from(value)),
            ElemwisePrecision::I32 => locals.l_i32.insert(pos, Line::cast_from(value)),
            ElemwisePrecision::Bool => locals.l_bool.insert(pos, Line::cast_from(value)),
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        _ => comptime![panic!("Can't write into inputs and scalars")],
    }
}

#[cube]
fn get_offset<C: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    tensor: &Tensor<Line<C>>,
    pos: u32,
    #[comptime] config: &ElemwiseConfig,
) -> u32 {
    match comptime![config.ref_layout] {
        Arg::Input(index, precision, _) => match comptime![precision] {
            ElemwisePrecision::F32 => {
                let layout = inputs.t_f32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::F16 => {
                let layout = inputs.t_f16.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::U32 => {
                let layout = inputs.t_u32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::I32 => {
                let layout = inputs.t_i32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        Arg::Output(index, precision, _) => match comptime![precision] {
            ElemwisePrecision::F32 => {
                let layout = outputs.t_f32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::F16 => {
                let layout = outputs.t_f16.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::U32 => {
                let layout = outputs.t_u32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::I32 => {
                let layout = outputs.t_i32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        _ => comptime![panic!("Invalid ref layout.")],
    }
}
