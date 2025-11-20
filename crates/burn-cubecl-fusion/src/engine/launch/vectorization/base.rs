use crate::{
    CubeFusionHandle,
    engine::launch::runner::{VectorizationAxis, VectorizationHandle},
};
use burn_fusion::stream::Context;
use burn_ir::{TensorId, TensorIr};
use cubecl::Runtime;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
pub enum Vect {
    Broadcasted,
    Aligned(u8),
}

impl Vect {
    pub fn line_size(&self) -> u8 {
        match self {
            Vect::Broadcasted => 1,
            Vect::Aligned(val) => *val,
        }
    }

    pub fn is_broadcast(&self) -> bool {
        matches!(self, Vect::Broadcasted)
    }

    pub fn limit_to_one(&self) -> Self {
        match self {
            Vect::Broadcasted => Vect::Broadcasted,
            Vect::Aligned(_) => Vect::Aligned(1),
        }
    }
}

#[derive(Default, Clone, Serialize, Deserialize, Debug)]
pub struct LineSizeOverrides {
    state: Option<BTreeMap<TensorId, Vec<u8>>>,
    default: Option<Vec<u8>>,
}

#[allow(unused)]
impl LineSizeOverrides {
    pub fn overrides(&mut self, tensor_id: &TensorId, line_sizes: Vec<u8>) {
        let map = match &mut self.state {
            Some(val) => val,
            None => {
                self.state = Some(BTreeMap::new());
                self.state.as_mut().unwrap()
            }
        };

        map.insert(*tensor_id, line_sizes);
    }
    pub fn overrides_default(&mut self, line_sizes: Vec<u8>) {
        self.default = Some(line_sizes);
    }

    pub fn mapping<R: Runtime>(&self, context: &Context<'_, CubeFusionHandle<R>>) -> Self {
        match &self.state {
            Some(state) => {
                let mut state_new = BTreeMap::new();

                for (k, v) in state.iter() {
                    let global = context.tensors.get(k).unwrap();
                    state_new.insert(global.id, v.clone());
                }

                Self {
                    state: Some(state_new),
                    default: self.default.clone(),
                }
            }
            None => Self {
                state: None,
                default: self.default.clone(),
            },
        }
    }

    pub fn tensor(&self, tensor_id: &TensorId) -> Option<&Vec<u8>> {
        let map = match &self.state {
            Some(val) => val,
            None => match &self.default {
                Some(val) => return Some(val),
                None => return None,
            },
        };

        match map.get(tensor_id) {
            Some(val) => Some(val),
            None => match &self.default {
                Some(val) => Some(val),
                None => None,
            },
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn vectorization_default<'a, R: Runtime>(
    vectorizations: &mut BTreeMap<TensorId, Vect>,
    inputs: impl Iterator<Item = VectorizationHandle<'a, R>>,
    outputs: impl Iterator<Item = &'a TensorIr>,
    reshaped: impl Iterator<Item = (&'a TensorIr, &'a TensorIr, bool)>,
    swapped: impl Iterator<Item = (&'a TensorIr, &'a TensorIr, bool, &'a (u32, u32))>,
    line_sizes: &[u8],
    overrides: &LineSizeOverrides,
    max: u8,
    axis: &VectorizationAxis,
) {
    let swapped: Vec<_> = swapped.collect();

    for input in inputs {
        if let Some((s, o, mr, dims)) = swapped
            .iter()
            .find(|(_s, o, _mr, _dims)| input.is_from_tensor(o.id))
        {
            let (handle, id) = match input {
                VectorizationHandle::NormalInput(handle, tensor_ir) => (handle, &tensor_ir.id),
                VectorizationHandle::QuantValues(..) => panic!("Can't be swapped"),
                VectorizationHandle::QuantParams(..) => panic!("Can't be swapped"),
            };
            let val = vectorization_swapped::<R>(
                handle,
                s,
                o,
                *mr,
                dims,
                max,
                axis,
                line_sizes,
                overrides.tensor(id),
            );
            multi_reads_vectorization_update(vectorizations, o.id, val);
        } else {
            match input {
                VectorizationHandle::NormalInput(handle, tensor_ir) => {
                    let val = vectorization_input(
                        handle,
                        tensor_ir,
                        axis,
                        line_sizes,
                        overrides.tensor(&tensor_ir.id),
                    );
                    vectorizations.insert(tensor_ir.id, val);
                }
                VectorizationHandle::QuantValues(handle, tensor_ir) => {
                    let val = vectorization_input(
                        handle,
                        tensor_ir,
                        axis,
                        line_sizes,
                        overrides.tensor(&tensor_ir.id),
                    );
                    let num_quants = match tensor_ir.dtype {
                        burn_tensor::DType::QFloat(quant_scheme) => quant_scheme.num_quants() as u8,
                        _ => panic!(""),
                    };
                    let val = match val {
                        Vect::Broadcasted => Vect::Aligned(1),
                        Vect::Aligned(val) => Vect::Aligned(val.div_ceil(num_quants)),
                    };
                    vectorizations.insert(tensor_ir.id, val);
                }
                VectorizationHandle::QuantParams(..) => {
                    // Doesn't have vectorization for now.
                }
            };
        }
    }

    for (reshaped, original, multi_reads) in reshaped {
        let val = vectorization_reshape(
            reshaped,
            original,
            multi_reads,
            axis,
            line_sizes,
            max,
            overrides.tensor(&original.id),
        );
        multi_reads_vectorization_update(vectorizations, original.id, val);
    }

    for tensor in outputs {
        let val = vectorization_output(tensor, axis, line_sizes, max, overrides.tensor(&tensor.id));
        vectorizations.insert(tensor.id, val);
    }
}

fn multi_reads_vectorization_update(
    vectorizations: &mut BTreeMap<TensorId, Vect>,
    original: TensorId,
    vect: Vect,
) {
    if let Some(ori_vect) = vectorizations.get(&original).cloned() {
        match ori_vect {
            Vect::Broadcasted => {
                // keep the original as is.
            }
            Vect::Aligned(ori) => match vect {
                Vect::Broadcasted => {
                    vectorizations.insert(original, Vect::Aligned(1));
                }
                Vect::Aligned(new) => {
                    let val = if new != ori { 1 } else { new };
                    vectorizations.insert(original, Vect::Aligned(val));
                }
            },
        };
    } else {
        vectorizations.insert(original, vect);
    }
}

// The default version uses the last dimension as vectorization axis and assumes a
// perpendicular contiguous line.
fn vectorization_input<R: Runtime>(
    handle: &CubeFusionHandle<R>,
    desc: &TensorIr,
    axis: &VectorizationAxis,
    line_sizes: &[u8],
    overrides: Option<&Vec<u8>>,
) -> Vect {
    let axis = axis.get(desc.id, || handle.strides.len() - 1);
    let shape_axis = desc.shape[axis];

    if shape_axis == 1 {
        return Vect::Broadcasted;
    }

    // Last dimension strides should be 1, otherwise vecX won't be contiguous.
    if handle.strides[axis] != 1 {
        return Vect::Aligned(1);
    }

    let inner = |s: u8| {
        // The last dimension should be a multiple of the vector size or broadcated.
        if shape_axis.is_multiple_of(s as usize) {
            return Some(Vect::Aligned(s));
        }
        None
    };

    match overrides {
        Some(vals) => {
            for s in vals {
                if let Some(val) = inner(*s) {
                    return val;
                }
            }
        }
        None => {
            for s in line_sizes {
                if let Some(val) = inner(*s) {
                    return val;
                }
            }
        }
    }

    Vect::Aligned(1)
}

fn vectorization_output(
    desc: &TensorIr,
    axis: &VectorizationAxis,
    line_sizes: &[u8],
    max: u8,
    overrides: Option<&Vec<u8>>,
) -> Vect {
    let axis = axis.get(desc.id, || desc.shape.rank() - 1);

    let inner = |s: u8| {
        // The dimension should be a multiple of the vector size.
        if desc.shape[axis].is_multiple_of(s as usize) && s <= max {
            return Some(Vect::Aligned(s));
        }

        None
    };
    match overrides {
        Some(val) => {
            for s in val {
                if let Some(val) = inner(*s) {
                    return val;
                }
            }
        }
        None => {
            for s in line_sizes {
                if let Some(val) = inner(*s) {
                    return val;
                }
            }
        }
    }

    Vect::Aligned(1)
}

fn vectorization_reshape(
    reshaped: &TensorIr,
    original: &TensorIr,
    multi_reads: bool,
    axis: &VectorizationAxis,
    line_sizes: &[u8],
    max: u8,
    overrides: Option<&Vec<u8>>,
) -> Vect {
    let axis = axis.get(reshaped.id, || reshaped.shape.rank() - 1);
    let reshape_shape_axis = reshaped.shape[axis];

    if !multi_reads && reshape_shape_axis == 1 {
        return Vect::Broadcasted;
    }

    // If the axis is not the last dim, didn't think of it, return Aligned(1) to be sure.
    if axis != reshaped.shape.rank() - 1 {
        return Vect::Aligned(1);
    }

    let original_shape_axis = original.shape[original.shape.rank() - 1];

    if original_shape_axis != reshape_shape_axis {
        return Vect::Aligned(1);
    }

    let inner = |s: u8| {
        if !multi_reads {
            // The last dimension should be a multiple of the vector size or broadcated.
            if reshape_shape_axis.is_multiple_of(s as usize) && s <= max {
                Some(Vect::Aligned(s))
            } else {
                None
            }
        } else {
            // Since the original tensor must share the same vectorization factor as the
            // reshaped tensor, they must have compatible shapes when both are access
            // independently.
            if reshape_shape_axis.is_multiple_of(s as usize)
                && original_shape_axis.is_multiple_of(s as usize)
                && s <= max
            {
                Some(Vect::Aligned(s))
            } else {
                None
            }
        }
    };

    match overrides {
        Some(val) => {
            for i in val {
                if let Some(vect) = inner(*i) {
                    return vect;
                }
            }
        }
        None => {
            for s in line_sizes {
                if let Some(vect) = inner(*s) {
                    return vect;
                }
            }
        }
    }

    Vect::Aligned(1)
}

#[allow(clippy::too_many_arguments)]
fn vectorization_swapped<R: Runtime>(
    handle: &CubeFusionHandle<R>,
    swapped: &TensorIr,
    original: &TensorIr,
    multi_reads: bool,
    dims: &(u32, u32),
    max: u8,
    axis: &VectorizationAxis,
    line_sizes: &[u8],
    overrides: Option<&Vec<u8>>,
) -> Vect {
    let axis = axis.get(swapped.id, || swapped.shape.rank() - 1);

    let swapped_axis = swapped.shape[axis];
    let shape_axis = original.shape[axis];

    let axis_index = axis;
    let dim_index = if dims.0 as usize == axis_index {
        dims.1 as usize
    } else if dims.1 as usize == axis_index {
        dims.0 as usize
    } else {
        axis_index
    };

    // Last dimension strides should be 1, otherwise vecX won't be contiguous.
    if multi_reads {
        if handle.strides[axis_index] != 1 {
            return Vect::Aligned(1);
        }
        if handle.strides[dim_index] != 1 {
            return Vect::Aligned(1);
        }
    } else if handle.strides[dim_index] != 1 {
        return Vect::Aligned(1);
    }

    if !multi_reads && swapped_axis == 1 {
        return Vect::Broadcasted;
    }

    let inner = |s: u8| {
        // The last dimension should be a multiple of the vector size or broadcated.
        if multi_reads {
            if swapped_axis.is_multiple_of(s as usize) && s <= max {
                return Some(Vect::Aligned(s));
            }
        } else if swapped_axis.is_multiple_of(s as usize)
            && shape_axis.is_multiple_of(s as usize)
            && s <= max
        {
            return Some(Vect::Aligned(s));
        }
        None
    };

    match overrides {
        Some(val) => {
            for s in val {
                if let Some(val) = inner(*s) {
                    return val;
                }
            }
        }
        None => {
            for s in line_sizes {
                if let Some(val) = inner(*s) {
                    return val;
                }
            }
        }
    }

    Vect::Aligned(1)
}
