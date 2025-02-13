use super::ir::*;
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
        Arg::Input(pos, precision, layout) => read_input(
            inputs, outputs, pos, ref_pos, layout, precision, config, None,
        ),
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
            original, shape, ..
        } => match comptime![original.as_ref().clone()] {
            Arg::Input(pos, precision, layout) => read_input(
                inputs,
                outputs,
                pos,
                ref_pos,
                layout,
                precision,
                config,
                comptime![Some(Transform::Reshape(shape))],
            ),
            _ => comptime![panic!("Only input can be reshaped")],
        },
        Arg::InputSwapDims { original, dims, .. } => match comptime![original.as_ref().clone()] {
            Arg::Input(pos, precision, layout) => read_input(
                inputs,
                outputs,
                pos,
                ref_pos,
                layout,
                precision,
                config,
                comptime![Some(Transform::SwapDim(dims.0, dims.1))],
            ),
            _ => comptime![panic!("Only input can be reshaped")],
        },
    }
}

#[cube]
pub fn read_scalar<C: CubePrimitive>(inputs: &GlobalArgs, #[comptime] arg: Arg) -> C {
    match arg {
        Arg::Scalar(pos, precision) => match comptime![precision] {
            ElemwisePrecision::F32 => C::cast_from(*inputs.s_f32.index(pos)),
            ElemwisePrecision::F16 => C::cast_from(*inputs.s_f16.index(pos)),
            ElemwisePrecision::BF16 => C::cast_from(*inputs.s_bf16.index(pos)),
            ElemwisePrecision::U64 => C::cast_from(*inputs.s_u64.index(pos)),
            ElemwisePrecision::U32 => C::cast_from(*inputs.s_u32.index(pos)),
            ElemwisePrecision::U16 => C::cast_from(*inputs.s_u16.index(pos)),
            ElemwisePrecision::U8 => C::cast_from(*inputs.s_u8.index(pos)),
            ElemwisePrecision::I64 => C::cast_from(*inputs.s_i64.index(pos)),
            ElemwisePrecision::I32 => C::cast_from(*inputs.s_i32.index(pos)),
            ElemwisePrecision::I16 => C::cast_from(*inputs.s_i16.index(pos)),
            ElemwisePrecision::I8 => C::cast_from(*inputs.s_i8.index(pos)),
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        _ => comptime![panic!("Not a scalar")],
    }
}

#[cube]
pub fn read_scalar_shape(inputs: &GlobalArgs, #[comptime] arg: Arg) -> u32 {
    match arg {
        Arg::ScalarShape(pos) => {
            let offset = comptime![inputs.s_u32.len() - pos - 1];
            *inputs.s_u32.index(offset)
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
    #[comptime] precision: ElemwisePrecision,
    #[comptime] config: &ElemwiseConfig,
    #[comptime] transform: Option<Transform>,
) -> Line<C> {
    match comptime![precision] {
        ElemwisePrecision::F32 => {
            let tensor = inputs.t_f32.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, transform)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::F16 => {
            let tensor = inputs.t_f16.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, transform)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::BF16 => {
            let tensor = inputs.t_bf16.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, transform)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::U64 => {
            let tensor = inputs.t_u64.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, transform)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::U32 => {
            let tensor = inputs.t_u32.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, transform)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::U16 => {
            let tensor = inputs.t_u16.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, transform)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::U8 => {
            let tensor = inputs.t_u8.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, transform)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::I64 => {
            let tensor = inputs.t_i64.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, transform)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::I32 => {
            let tensor = inputs.t_i32.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, transform)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::I16 => {
            let tensor = inputs.t_i16.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, transform)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::I8 => {
            let tensor = inputs.t_i8.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, transform)
                }
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
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::F16 => {
            let tensor = outputs.t_f16.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::BF16 => {
            let tensor = outputs.t_bf16.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::U64 => {
            let tensor = outputs.t_u64.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::U32 => {
            let tensor = outputs.t_u32.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::U16 => {
            let tensor = outputs.t_u16.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::U8 => {
            let tensor = outputs.t_u8.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::I64 => {
            let tensor = outputs.t_i64.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::I32 => {
            let tensor = outputs.t_i32.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::I16 => {
            let tensor = outputs.t_i16.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                }
            };
            Line::cast_from(tensor[offset])
        }
        ElemwisePrecision::I8 => {
            let tensor = outputs.t_i8.index(pos);
            let offset = match layout {
                LayoutInfo::SameAsRef => ref_pos,
                LayoutInfo::IsRef => ref_pos,
                LayoutInfo::Unknown => {
                    get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                }
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
                    LayoutInfo::Unknown => {
                        get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                    }
                };
                let tensor = outputs.t_f32.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::F16 => {
                let tensor = outputs.t_f16.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => {
                        get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                    }
                };
                let tensor = outputs.t_f16.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::BF16 => {
                let tensor = outputs.t_bf16.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => {
                        get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                    }
                };
                let tensor = outputs.t_bf16.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::U64 => {
                let tensor = outputs.t_u64.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => {
                        get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                    }
                };
                let tensor = outputs.t_u64.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::U32 => {
                let tensor = outputs.t_u32.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => {
                        get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                    }
                };
                let tensor = outputs.t_u32.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::U16 => {
                let tensor = outputs.t_u16.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => {
                        get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                    }
                };
                let tensor = outputs.t_u16.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::U8 => {
                let tensor = outputs.t_u8.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => {
                        get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                    }
                };
                let tensor = outputs.t_u8.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::I64 => {
                let tensor = outputs.t_i64.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => {
                        get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                    }
                };
                let tensor = outputs.t_i64.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::I32 => {
                let tensor = outputs.t_i32.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => {
                        get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                    }
                };
                let tensor = outputs.t_i32.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::I16 => {
                let tensor = outputs.t_i16.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => {
                        get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                    }
                };
                let tensor = outputs.t_i16.index_mut(pos);
                tensor[offset] = Line::cast_from(value);
            }
            ElemwisePrecision::I8 => {
                let tensor = outputs.t_i8.index(pos);
                let offset = match layout {
                    LayoutInfo::SameAsRef => ref_pos,
                    LayoutInfo::IsRef => ref_pos,
                    LayoutInfo::Unknown => {
                        get_offset(inputs, outputs, tensor, ref_pos, None, config, None)
                    }
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
pub(crate) fn global_offset(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    index: u32,
    #[comptime] arg: Arg,
    #[comptime] range: Option<(u32, u32)>,
    #[comptime] config: &ElemwiseConfig,
) -> u32 {
    match arg {
        Arg::Input(pos, precision, _layout) => match precision {
            ElemwisePrecision::F32 => get_offset(
                inputs,
                outputs,
                inputs.t_f32.index(pos),
                index,
                range,
                config,
                None,
            ),
            ElemwisePrecision::F16 => get_offset(
                inputs,
                outputs,
                inputs.t_f16.index(pos),
                index,
                range,
                config,
                None,
            ),
            ElemwisePrecision::BF16 => get_offset(
                inputs,
                outputs,
                inputs.t_bf16.index(pos),
                index,
                range,
                config,
                None,
            ),
            ElemwisePrecision::I64 => get_offset(
                inputs,
                outputs,
                inputs.t_i64.index(pos),
                index,
                range,
                config,
                None,
            ),
            ElemwisePrecision::I32 => get_offset(
                inputs,
                outputs,
                inputs.t_i32.index(pos),
                index,
                range,
                config,
                None,
            ),
            ElemwisePrecision::I16 => get_offset(
                inputs,
                outputs,
                inputs.t_i16.index(pos),
                index,
                range,
                config,
                None,
            ),
            ElemwisePrecision::I8 => get_offset(
                inputs,
                outputs,
                inputs.t_i8.index(pos),
                index,
                range,
                config,
                None,
            ),
            ElemwisePrecision::U64 => get_offset(
                inputs,
                outputs,
                inputs.t_u64.index(pos),
                index,
                range,
                config,
                None,
            ),
            ElemwisePrecision::U32 => get_offset(
                inputs,
                outputs,
                inputs.t_u32.index(pos),
                index,
                range,
                config,
                None,
            ),
            ElemwisePrecision::U16 => get_offset(
                inputs,
                outputs,
                inputs.t_u16.index(pos),
                index,
                range,
                config,
                None,
            ),
            ElemwisePrecision::U8 => get_offset(
                inputs,
                outputs,
                inputs.t_u8.index(pos),
                index,
                range,
                config,
                None,
            ),
            ElemwisePrecision::Bool => comptime!(panic!(
                "Should be resolved to the correct bool type used by the backend"
            )),
        },
        _ => todo!(),
    }
}

#[cube]
fn get_offset<C: CubePrimitive>(
    inputs: &GlobalArgs,
    outputs: &GlobalArgs,
    tensor: &Tensor<Line<C>>,
    pos: u32,
    #[comptime] range: Option<(u32, u32)>,
    #[comptime] config: &ElemwiseConfig,
    #[comptime] transform: Option<Transform>,
) -> u32 {
    match comptime![config.ref_layout.clone()] {
        Arg::Input(index, precision, _) => match comptime![precision] {
            ElemwisePrecision::F32 => {
                let layout = inputs.t_f32.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::F16 => {
                let layout = inputs.t_f16.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::BF16 => {
                let layout = inputs.t_bf16.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::U64 => {
                let layout = inputs.t_u64.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::U32 => {
                let layout = inputs.t_u32.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::U16 => {
                let layout = inputs.t_u16.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::U8 => {
                let layout = inputs.t_u8.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::I64 => {
                let layout = inputs.t_i64.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::I32 => {
                let layout = inputs.t_i32.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::I16 => {
                let layout = inputs.t_i16.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::I8 => {
                let layout = inputs.t_i8.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        Arg::Output(index, precision, _) => match comptime![precision] {
            ElemwisePrecision::F32 => {
                let layout = outputs.t_f32.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::F16 => {
                let layout = outputs.t_f16.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::BF16 => {
                let layout = outputs.t_bf16.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::U64 => {
                let layout = outputs.t_u64.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::U32 => {
                let layout = outputs.t_u32.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::U16 => {
                let layout = outputs.t_u16.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::U8 => {
                let layout = outputs.t_u8.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::I64 => {
                let layout = outputs.t_i64.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::I32 => {
                let layout = outputs.t_i32.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::I16 => {
                let layout = outputs.t_i16.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            ElemwisePrecision::I8 => {
                let layout = outputs.t_i8.index(index);
                index_offset_with_layout(inputs, tensor, layout, pos, range, config.rank, transform)
            }
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        _ => comptime![panic!("Invalid ref layout.")],
    }
}

#[cube]
pub fn global_line_size(
    global: &GlobalArgs,
    #[comptime] pos: u32,
    #[comptime] precision: ElemwisePrecision,
) -> u32 {
    u32::cast_from(match comptime![precision] {
        ElemwisePrecision::F32 => {
            let tensor = global.t_f32.index(pos);
            tensor.line_size()
        }
        ElemwisePrecision::F16 => {
            let tensor = global.t_f16.index(pos);
            tensor.line_size()
        }
        ElemwisePrecision::BF16 => {
            let tensor = global.t_bf16.index(pos);
            tensor.line_size()
        }
        ElemwisePrecision::U64 => {
            let tensor = global.t_u64.index(pos);
            tensor.line_size()
        }
        ElemwisePrecision::U32 => {
            let tensor = global.t_u32.index(pos);
            tensor.line_size()
        }
        ElemwisePrecision::U16 => {
            let tensor = global.t_u16.index(pos);
            tensor.line_size()
        }
        ElemwisePrecision::U8 => {
            let tensor = global.t_u8.index(pos);
            tensor.line_size()
        }
        ElemwisePrecision::I64 => {
            let tensor = global.t_i64.index(pos);
            tensor.line_size()
        }
        ElemwisePrecision::I32 => {
            let tensor = global.t_i32.index(pos);
            tensor.line_size()
        }
        ElemwisePrecision::I16 => {
            let tensor = global.t_i16.index(pos);
            tensor.line_size()
        }
        ElemwisePrecision::I8 => {
            let tensor = global.t_i8.index(pos);
            tensor.line_size()
        }
        _ => comptime![panic!("Unsupported precision {precision:?}")],
    })
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

/// Returns the offset of the tensor corresponding to the layout tensor.
#[cube]
fn index_offset_with_layout<N: CubePrimitive, L: CubePrimitive>(
    inputs: &GlobalArgs,
    tensor: &Tensor<Line<N>>,
    layout: &Tensor<Line<L>>,
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
            let index = reshaped_index(inputs, layout, index, rank, shape);
            reshaped_index_to_original_index(tensor, index, rank)
        }
        Some(Transform::SwapDim(dim1, dim2)) => {
            let (start, end) = comptime! {match range {
                Some(range) => range,
                None => (0u32, rank),
            }};

            let offset_ref = index * layout.line_size();
            let mut offset = 0u32;

            #[unroll]
            for i in start..end {
                let index = comptime![swap_dims_transform(&i, (dim1, dim2))];
                let ogwl = offset_ref / layout.stride(i);
                offset += ogwl % tensor.shape(index) * tensor.stride(index);
            }

            offset / tensor.line_size()
        }
        None => {
            let (start, end) = comptime! {match range {
                Some(range) => range,
                None => (0u32, rank),
            }};

            let offset_ref = index * layout.line_size();
            let mut offset = 0u32;

            for i in start..end {
                let ogwl = offset_ref / layout.stride(i);
                offset += ogwl % tensor.shape(i) * tensor.stride(i);
            }

            offset / tensor.line_size()
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
fn reshaped_index<N: CubePrimitive>(
    inputs: &GlobalArgs,
    layout: &Tensor<Line<N>>,
    index: u32,
    #[comptime] rank: u32,
    #[comptime] shape: Sequence<Arg>,
) -> u32 {
    let index = index * layout.line_size();

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
