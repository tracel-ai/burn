use super::ir::*;
use cubecl::{linalg::tensor::index_offset_with_layout, prelude::*};

#[cube]
/// Read the value from the [arg](Arg) and cast it to the generic cube primitive.
pub fn read<C: CubePrimitive>(
    inputs: &FusionArgs,
    outputs: &FusionArgs,
    locals: &FusionLocals,
    ref_pos: u32,
    #[comptime] arg: Arg,
    #[comptime] config: &FusionConfig,
) -> Line<C> {
    match arg {
        Arg::Input(pos, precision) => match comptime![precision] {
            OpPrecision::F32 => {
                let tensor = inputs.t_f32.index(pos);
                let offset = get_offset(inputs, outputs, tensor, ref_pos, config);
                Line::cast_from(tensor[offset])
            }
            OpPrecision::F16 => {
                let tensor = inputs.t_f16.index(pos);
                let offset = get_offset(inputs, outputs, tensor, ref_pos, config);
                Line::cast_from(tensor[offset])
            }
            OpPrecision::U32 => {
                let tensor = inputs.t_u32.index(pos);
                let offset = get_offset(inputs, outputs, tensor, ref_pos, config);
                Line::cast_from(tensor[offset])
            }
            OpPrecision::I32 => {
                let tensor = inputs.t_i32.index(pos);
                let offset = get_offset(inputs, outputs, tensor, ref_pos, config);
                Line::cast_from(tensor[offset])
            }
            _ => comptime![panic!("Unsupported")],
        },
        Arg::Output(pos, precision) => match comptime![precision] {
            OpPrecision::F32 => {
                let tensor = outputs.t_f32.index(pos);
                let offset = get_offset(inputs, outputs, tensor, ref_pos, config);
                Line::cast_from(tensor[offset])
            }
            OpPrecision::F16 => {
                let tensor = outputs.t_f16.index(pos);
                let offset = get_offset(inputs, outputs, tensor, ref_pos, config);
                Line::cast_from(tensor[offset])
            }
            OpPrecision::U32 => {
                let tensor = outputs.t_u32.index(pos);
                let offset = get_offset(inputs, outputs, tensor, ref_pos, config);
                Line::cast_from(tensor[offset])
            }
            OpPrecision::I32 => {
                let tensor = outputs.t_i32.index(pos);
                let offset = get_offset(inputs, outputs, tensor, ref_pos, config);
                Line::cast_from(tensor[offset])
            }
            _ => comptime![panic!("Unsupported")],
        },
        Arg::Local(pos, precision) => match comptime![precision] {
            OpPrecision::F32 => Line::cast_from(*locals.l_f32.index(pos)),
            OpPrecision::F16 => Line::cast_from(*locals.l_f16.index(pos)),
            OpPrecision::U32 => Line::cast_from(*locals.l_u32.index(pos)),
            OpPrecision::I32 => Line::cast_from(*locals.l_i32.index(pos)),
            _ => comptime![panic!("Can't write into inputs or scalars")],
        },
        Arg::Scalar(pos, precision) => match comptime![precision] {
            OpPrecision::F32 => Line::cast_from(*inputs.s_f32.index(pos)),
            OpPrecision::F16 => Line::cast_from(*inputs.s_f16.index(pos)),
            OpPrecision::U32 => Line::cast_from(*inputs.s_u32.index(pos)),
            OpPrecision::I32 => Line::cast_from(*inputs.s_i32.index(pos)),
            OpPrecision::BF16 => comptime![panic!("Can't write into inputs or scalars")],
            _ => comptime![panic!("Can't write into inputs or scalars")],
        },
        Arg::Literal(val, _precision) => Line::cast_from(val.runtime()),
    }
}

#[cube]
/// Read the value from the [arg](Arg) and cast it to the generic cube primitive.
pub fn write<C: CubePrimitive>(
    inputs: &FusionArgs,
    outputs: &mut FusionArgs,
    locals: &mut FusionLocals,
    ref_pos: u32,
    value: Line<C>,
    #[comptime] arg: Arg,
    #[comptime] config: &FusionConfig,
) {
    match arg {
        Arg::Output(pos, precision) => match comptime![precision] {
            OpPrecision::F32 => {
                let tensor = outputs.t_f32.index(pos);
                let offset = get_offset(inputs, outputs, tensor, ref_pos, config);
                let tensor = outputs.t_f32.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            OpPrecision::F16 => {
                let tensor = outputs.t_f16.index(pos);
                let offset = get_offset(inputs, outputs, tensor, ref_pos, config);
                let tensor = outputs.t_f16.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            OpPrecision::U32 => {
                let tensor = outputs.t_u32.index(pos);
                let offset = get_offset(inputs, outputs, tensor, ref_pos, config);
                let tensor = outputs.t_u32.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            OpPrecision::I32 => {
                let tensor = outputs.t_i32.index(pos);
                let offset = get_offset(inputs, outputs, tensor, ref_pos, config);
                let tensor = outputs.t_i32.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            _ => comptime![panic!("Unsupported")],
        },
        Arg::Local(pos, precision) => match comptime![precision] {
            OpPrecision::F32 => locals.l_f32.insert(pos, Line::cast_from(value)),
            OpPrecision::F16 => locals.l_f16.insert(pos, Line::cast_from(value)),
            OpPrecision::U32 => locals.l_u32.insert(pos, Line::cast_from(value)),
            OpPrecision::I32 => locals.l_i32.insert(pos, Line::cast_from(value)),
            _ => comptime![panic!("Unsupported")],
        },
        _ => comptime![panic!("Can't write into inputs and scalars")],
    }
}

#[cube]
pub fn get_offset<C: CubePrimitive>(
    inputs: &FusionArgs,
    outputs: &FusionArgs,
    tensor: &Tensor<Line<C>>,
    pos: u32,
    #[comptime] config: &FusionConfig,
) -> u32 {
    match comptime![config.ref_layout.arg] {
        Arg::Input(index, precision) => match comptime![precision] {
            OpPrecision::F32 => {
                let layout = inputs.t_f32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            OpPrecision::F16 => {
                let layout = inputs.t_f16.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            OpPrecision::U32 => {
                let layout = inputs.t_u32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            OpPrecision::I32 => {
                let layout = inputs.t_i32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            _ => comptime![panic!("Unsupported")],
        },
        Arg::Output(index, precision) => match comptime![precision] {
            OpPrecision::F32 => {
                let layout = outputs.t_f32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            OpPrecision::F16 => {
                let layout = outputs.t_f16.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            OpPrecision::U32 => {
                let layout = outputs.t_u32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            OpPrecision::I32 => {
                let layout = outputs.t_i32.index(index);
                index_offset_with_layout(tensor, layout, pos, 0, config.rank, false)
            }
            _ => comptime![panic!("Unsupported")],
        },
        _ => comptime![panic!("Unsupported")],
    }
}
