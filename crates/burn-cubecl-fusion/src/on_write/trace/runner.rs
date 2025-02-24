use super::{
    super::ir::{ElemwiseConfig, GlobalArgsLaunch},
    Vect,
};
use crate::CubeFusionHandle;
use burn_ir::{TensorId, TensorIr};
use cubecl::{ir::Elem, prelude::*};
use std::collections::BTreeMap;

/// A trace runner is responsible for determining the vectorization factor as well as launching
/// a kernel based on global [inputs](GlobalArgsLaunch) and [outputs](GlobalArgsLaunch)
/// with a provided [element wise config](ElemwiseConfig).
pub trait TraceRunner<R: Runtime> {
    /// The error that might happen while running the trace.
    type Error;

    /// Run the trace.
    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        config: &'a ElemwiseConfig,
    ) -> Result<(), Self::Error>;

    /// The vectorization factor for all inputs and outputs.
    fn vectorization<'a>(
        vectorizations: &mut BTreeMap<TensorId, Vect>,
        handles_inputs: impl Iterator<Item = &'a CubeFusionHandle<R>>,
        inputs: impl Iterator<Item = &'a TensorIr>,
        outputs: impl Iterator<Item = &'a TensorIr>,
        reshaped: impl Iterator<Item = (&'a TensorIr, &'a TensorIr, bool)>,
        swapped: impl Iterator<Item = (&'a TensorIr, &'a TensorIr, bool, &'a (u32, u32))>,
        ref_elem: &Elem,
    ) {
        vectorization_default(
            vectorizations,
            handles_inputs,
            inputs,
            outputs,
            reshaped,
            swapped,
            ref_elem,
        )
    }
}

fn vectorization_default<'a, R: Runtime>(
    vectorizations: &mut BTreeMap<TensorId, Vect>,
    handles_inputs: impl Iterator<Item = &'a CubeFusionHandle<R>>,
    inputs: impl Iterator<Item = &'a TensorIr>,
    outputs: impl Iterator<Item = &'a TensorIr>,
    reshaped: impl Iterator<Item = (&'a TensorIr, &'a TensorIr, bool)>,
    swapped: impl Iterator<Item = (&'a TensorIr, &'a TensorIr, bool, &'a (u32, u32))>,
    // Smallest element type that can be vectorized.
    ref_elem: &Elem,
) {
    let swapped: Vec<_> = swapped.collect();

    // The default version uses the last dimension as vectorization axis and assumes a
    // perpendicular contiguous line.
    let vectorization_input = |handle: &CubeFusionHandle<R>, desc: &TensorIr| {
        let rank = handle.strides.len();
        let shape_axis = desc.shape[rank - 1];

        if shape_axis == 1 {
            return Vect::Broadcated;
        }

        // Last dimension strides should be 1, otherwise vecX won't be contiguous.
        if handle.strides[rank - 1] != 1 {
            return Vect::Aligned(1);
        }

        for s in R::line_size_elem(ref_elem) {
            // The last dimension should be a multiple of the vector size or broadcated.
            if shape_axis % s as usize == 0 {
                return Vect::Aligned(s);
            }
        }

        Vect::Aligned(1)
    };

    let vectorization_output = |desc: &TensorIr| {
        let rank = desc.shape.len();

        for s in R::line_size_elem(ref_elem) {
            // The last dimension should be a multiple of the vector size.
            if desc.shape[rank - 1] % s as usize == 0 {
                return Vect::Aligned(s);
            }
        }

        Vect::Aligned(1)
    };

    let vectorization_reshape = |reshaped: &TensorIr, original: &TensorIr, multi_reads: bool| {
        let reshape_axis = reshaped.shape[reshaped.shape.len() - 1];
        let shape_axis = original.shape[original.shape.len() - 1];

        if !multi_reads && reshape_axis == 1 {
            return Vect::Broadcated;
        }

        for s in R::line_size_elem(ref_elem) {
            if !multi_reads {
                // The last dimension should be a multiple of the vector size or broadcated.
                if reshape_axis % s as usize == 0 {
                    return Vect::Aligned(s);
                }
            } else {
                // Since the original tensor must share the same vectorization factor as the
                // reshaped tensor, they must have compatible shapes when both are access
                // independently.
                if reshape_axis % s as usize == 0 && shape_axis % s as usize == 0 {
                    return Vect::Aligned(s);
                }
            }
        }

        Vect::Aligned(1)
    };

    let vectorization_swapped = |handle: &CubeFusionHandle<R>,
                                 swapped: &TensorIr,
                                 original: &TensorIr,
                                 multi_reads: bool,
                                 dims: &(u32, u32)| {
        let swapped_axis = swapped.shape[swapped.shape.len() - 1];
        let shape_axis = original.shape[original.shape.len() - 1];

        let last_dim_index = handle.strides.len() - 1;
        let dim_index = if dims.0 as usize == last_dim_index {
            dims.1 as usize
        } else if dims.1 as usize == last_dim_index {
            dims.0 as usize
        } else {
            last_dim_index
        };

        // Last dimension strides should be 1, otherwise vecX won't be contiguous.
        if multi_reads {
            if handle.strides[last_dim_index] != 1 {
                return Vect::Aligned(1);
            }
            if handle.strides[dim_index] != 1 {
                return Vect::Aligned(1);
            }
        } else if handle.strides[dim_index] != 1 {
            return Vect::Aligned(1);
        }

        if !multi_reads && swapped_axis == 1 {
            return Vect::Broadcated;
        }

        for s in R::line_size_elem(ref_elem) {
            // The last dimension should be a multiple of the vector size or broadcated.
            if multi_reads {
                if swapped_axis % s as usize == 0 {
                    return Vect::Aligned(s);
                }
            } else if swapped_axis % s as usize == 0 && shape_axis % s as usize == 0 {
                return Vect::Aligned(s);
            }
        }

        Vect::Aligned(1)
    };

    for (handle, tensor) in handles_inputs.zip(inputs) {
        if let Some((s, o, mr, dims)) = swapped.iter().find(|(_s, o, _mr, _dims)| o.id == tensor.id)
        {
            let val = vectorization_swapped(handle, s, o, *mr, dims);
            multi_reads_vectorization_update(vectorizations, o.id, s.id, val);
        } else {
            let val = vectorization_input(handle, tensor);
            vectorizations.insert(tensor.id, val);
        }
    }

    for tensor in outputs {
        let val = vectorization_output(tensor);
        vectorizations.insert(tensor.id, val);
    }

    for (reshaped, original, multi_reads) in reshaped {
        let val = vectorization_reshape(reshaped, original, multi_reads);
        multi_reads_vectorization_update(vectorizations, original.id, reshaped.id, val);
    }
}

fn multi_reads_vectorization_update(
    vectorizations: &mut BTreeMap<TensorId, Vect>,
    original: TensorId,
    view: TensorId,
    vect: Vect,
) {
    if let Some(ori_vect) = vectorizations.get(&original).cloned() {
        match ori_vect {
            Vect::Broadcated => {
                // keep the original as is.
                vectorizations.insert(view, vect.limit_to_one());
            }
            Vect::Aligned(ori) => match vect {
                Vect::Broadcated => {
                    vectorizations.insert(original, Vect::Aligned(1));
                    vectorizations.insert(view, vect.limit_to_one());
                }
                Vect::Aligned(new) => {
                    let val = if new != ori { 1 } else { new };
                    vectorizations.insert(original, Vect::Aligned(val));
                    vectorizations.insert(view, Vect::Aligned(val));
                }
            },
        };
    } else {
        vectorizations.insert(original, vect);
        vectorizations.insert(view, vect);
    }
}
