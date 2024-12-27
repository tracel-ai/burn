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
        Arg::Input(pos, precision, layout) => {
            read_input(inputs, outputs, pos, ref_pos, layout, precision, config)
        }
        Arg::Output(pos, precision, layout) => {
            read_output(inputs, outputs, pos, ref_pos, layout, precision, config)
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
        Arg::Scalar(pos, precision) => match comptime![precision] {
            ElemwisePrecision::F32 => Line::cast_from(*inputs.s_f32.index(pos)),
            ElemwisePrecision::F16 => Line::cast_from(*inputs.s_f16.index(pos)),
            ElemwisePrecision::BF16 => Line::cast_from(*inputs.s_bf16.index(pos)),
            ElemwisePrecision::U64 => Line::cast_from(*inputs.s_u64.index(pos)),
            ElemwisePrecision::U32 => Line::cast_from(*inputs.s_u32.index(pos)),
            ElemwisePrecision::U16 => Line::cast_from(*inputs.s_u16.index(pos)),
            ElemwisePrecision::U8 => Line::cast_from(*inputs.s_u8.index(pos)),
            ElemwisePrecision::I64 => Line::cast_from(*inputs.s_i64.index(pos)),
            ElemwisePrecision::I32 => Line::cast_from(*inputs.s_i32.index(pos)),
            ElemwisePrecision::I16 => Line::cast_from(*inputs.s_i16.index(pos)),
            ElemwisePrecision::I8 => Line::cast_from(*inputs.s_i8.index(pos)),
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        Arg::Literal(val, _precision) => Line::cast_from(val.runtime()),
    }
}

#[cube]
pub fn read_input<C: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    #[comptime] pos: u32,
    ref_pos: u32,
    #[comptime] layout: LayoutInfo,
    #[comptime] precision: ElemwisePrecision,
    #[comptime] config: &ElemwiseConfig,
) -> Line<C> {
    match comptime![precision] {
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
        ElemwisePrecision::BF16 => {
            let tensor = inputs.t_bf16.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::U64 => {
            let tensor = inputs.t_u64.index(pos);
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
        ElemwisePrecision::U16 => {
            let tensor = inputs.t_u16.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::U8 => {
            let tensor = inputs.t_u8.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::I64 => {
            let tensor = inputs.t_i64.index(pos);
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
        ElemwisePrecision::I16 => {
            let tensor = inputs.t_i16.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::I8 => {
            let tensor = inputs.t_i8.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
            };
            Line::cast_from(tensor[offset])
        }
        _ => comptime![panic!("Unsupported precision {precision:?}")],
    }
}

#[cube]
pub fn read_output<C: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    pos: u32,
    ref_pos: u32,
    #[comptime] layout: LayoutInfo,
    #[comptime] precision: ElemwisePrecision,
    #[comptime] config: &ElemwiseConfig,
) -> Line<C> {
    match comptime![precision] {
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
        ElemwisePrecision::BF16 => {
            let tensor = outputs.t_bf16.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::U64 => {
            let tensor = outputs.t_u64.index(pos);
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
        ElemwisePrecision::U16 => {
            let tensor = outputs.t_u16.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::U8 => {
            let tensor = outputs.t_u8.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::I64 => {
            let tensor = outputs.t_i64.index(pos);
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
        ElemwisePrecision::I16 => {
            let tensor = outputs.t_i16.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::I8 => {
            let tensor = outputs.t_i8.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
            };
            Line::cast_from(tensor[offset])
        }
        _ => comptime![panic!("Unsupported precision {precision:?}")],
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
            ElemwisePrecision::BF16 => {
                let tensor = outputs.t_bf16.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                let tensor = outputs.t_bf16.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::U64 => {
                let tensor = outputs.t_u64.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                let tensor = outputs.t_u64.index_mut(pos);
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
            ElemwisePrecision::U16 => {
                let tensor = outputs.t_u16.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                let tensor = outputs.t_u16.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::U8 => {
                let tensor = outputs.t_u8.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                let tensor = outputs.t_u8.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::I64 => {
                let tensor = outputs.t_i64.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                let tensor = outputs.t_i64.index_mut(pos);
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
            ElemwisePrecision::I16 => {
                let tensor = outputs.t_i16.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                let tensor = outputs.t_i16.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::I8 => {
                let tensor = outputs.t_i8.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => get_offset(inputs, outputs, tensor, ref_pos, config),
                };
                let tensor = outputs.t_i8.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
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
            ElemwisePrecision::BF16 => {
                let layout = inputs.t_bf16.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::U64 => {
                let layout = inputs.t_u64.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::U32 => {
                let layout = inputs.t_u32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::U16 => {
                let layout = inputs.t_u16.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::U8 => {
                let layout = inputs.t_u8.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::I64 => {
                let layout = inputs.t_i64.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::I32 => {
                let layout = inputs.t_i32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::I16 => {
                let layout = inputs.t_i16.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::I8 => {
                let layout = inputs.t_i8.index(index);
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
            ElemwisePrecision::BF16 => {
                let layout = outputs.t_bf16.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::U64 => {
                let layout = outputs.t_u64.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::U32 => {
                let layout = outputs.t_u32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::U16 => {
                let layout = outputs.t_u16.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::U8 => {
                let layout = outputs.t_u8.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::I64 => {
                let layout = outputs.t_i64.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::I32 => {
                let layout = outputs.t_i32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::I16 => {
                let layout = outputs.t_i16.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            ElemwisePrecision::I8 => {
                let layout = outputs.t_i8.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        _ => comptime![panic!("Invalid ref layout.")],
    }
}
#[cube]
pub fn global_rank(
    global: &GlobalArgs,
    #[comptime] pos: u32,
    #[comptime] precision: ElemwisePrecision,
) -> u32 {
    match comptime![precision] {
        ElemwisePrecision::F32 => {
            let tensor = global.t_f32.index(pos);
            tensor.rank()
        }
        ElemwisePrecision::F16 => {
            let tensor = global.t_f16.index(pos);
            tensor.rank()
        }
        ElemwisePrecision::BF16 => {
            let tensor = global.t_bf16.index(pos);
            tensor.rank()
        }
        ElemwisePrecision::U64 => {
            let tensor = global.t_u64.index(pos);
            tensor.rank()
        }
        ElemwisePrecision::U32 => {
            let tensor = global.t_u32.index(pos);
            tensor.rank()
        }
        ElemwisePrecision::U16 => {
            let tensor = global.t_u16.index(pos);
            tensor.rank()
        }
        ElemwisePrecision::U8 => {
            let tensor = global.t_u8.index(pos);
            tensor.rank()
        }
        ElemwisePrecision::I64 => {
            let tensor = global.t_i64.index(pos);
            tensor.rank()
        }
        ElemwisePrecision::I32 => {
            let tensor = global.t_i32.index(pos);
            tensor.rank()
        }
        ElemwisePrecision::I16 => {
            let tensor = global.t_i16.index(pos);
            tensor.rank()
        }
        ElemwisePrecision::I8 => {
            let tensor = global.t_i8.index(pos);
            tensor.rank()
        }
        _ => comptime![panic!("Unsupported precision {precision:?}")],
    }
}
#[cube]
pub fn global_shape(
    global: &GlobalArgs,
    dim: u32,
    #[comptime] pos: u32,
    #[comptime] precision: ElemwisePrecision,
) -> u32 {
    match comptime![precision] {
        ElemwisePrecision::F32 => {
            let tensor = global.t_f32.index(pos);
            tensor.shape(dim)
        }
        ElemwisePrecision::F16 => {
            let tensor = global.t_f16.index(pos);
            tensor.shape(dim)
        }
        ElemwisePrecision::BF16 => {
            let tensor = global.t_bf16.index(pos);
            tensor.shape(dim)
        }
        ElemwisePrecision::U64 => {
            let tensor = global.t_u64.index(pos);
            tensor.shape(dim)
        }
        ElemwisePrecision::U32 => {
            let tensor = global.t_u32.index(pos);
            tensor.shape(dim)
        }
        ElemwisePrecision::U16 => {
            let tensor = global.t_u16.index(pos);
            tensor.shape(dim)
        }
        ElemwisePrecision::U8 => {
            let tensor = global.t_u8.index(pos);
            tensor.shape(dim)
        }
        ElemwisePrecision::I64 => {
            let tensor = global.t_i64.index(pos);
            tensor.shape(dim)
        }
        ElemwisePrecision::I32 => {
            let tensor = global.t_i32.index(pos);
            tensor.shape(dim)
        }
        ElemwisePrecision::I16 => {
            let tensor = global.t_i16.index(pos);
            tensor.shape(dim)
        }
        ElemwisePrecision::I8 => {
            let tensor = global.t_i8.index(pos);
            tensor.shape(dim)
        }
        _ => comptime![panic!("Unsupported precision {precision:?}")],
    }
}

#[cube]
pub fn global_stride(
    global: &GlobalArgs,
    dim: u32,
    #[comptime] pos: u32,
    #[comptime] precision: ElemwisePrecision,
) -> u32 {
    match comptime![precision] {
        ElemwisePrecision::F32 => {
            let tensor = global.t_f32.index(pos);
            tensor.stride(dim)
        }
        ElemwisePrecision::F16 => {
            let tensor = global.t_f16.index(pos);
            tensor.stride(dim)
        }
        ElemwisePrecision::BF16 => {
            let tensor = global.t_bf16.index(pos);
            tensor.stride(dim)
        }
        ElemwisePrecision::U64 => {
            let tensor = global.t_u64.index(pos);
            tensor.stride(dim)
        }
        ElemwisePrecision::U32 => {
            let tensor = global.t_u32.index(pos);
            tensor.stride(dim)
        }
        ElemwisePrecision::U16 => {
            let tensor = global.t_u16.index(pos);
            tensor.stride(dim)
        }
        ElemwisePrecision::U8 => {
            let tensor = global.t_u8.index(pos);
            tensor.stride(dim)
        }
        ElemwisePrecision::I64 => {
            let tensor = global.t_i64.index(pos);
            tensor.stride(dim)
        }
        ElemwisePrecision::I32 => {
            let tensor = global.t_i32.index(pos);
            tensor.stride(dim)
        }
        ElemwisePrecision::I16 => {
            let tensor = global.t_i16.index(pos);
            tensor.stride(dim)
        }
        ElemwisePrecision::I8 => {
            let tensor = global.t_i8.index(pos);
            tensor.stride(dim)
        }
        _ => comptime![panic!("Unsupported precision {precision:?}")],
    }
}
