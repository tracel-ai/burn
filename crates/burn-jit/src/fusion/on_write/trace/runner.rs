use super::super::ir::{ElemwiseConfig, GlobalArgsLaunch};
use crate::{fusion::JitFusionHandle, JitRuntime};
use burn_tensor::repr::{TensorDescription, TensorId};
use cubecl::prelude::*;
use std::collections::BTreeMap;

/// A trace runner is responsible for determining the vectorization factor as well as launching
/// a kernel based on global [inputs](GlobalArgsLaunch) and [outputs](GlobalArgsLaunch)
/// with a provided [element wise config](ElemwiseConfig).
pub trait TraceRunner<R: JitRuntime> {
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
        vectorizations: &mut BTreeMap<TensorId, u8>,
        handles_inputs: impl Iterator<Item = &'a JitFusionHandle<R>>,
        inputs: impl Iterator<Item = &'a TensorDescription>,
        outputs: impl Iterator<Item = &'a TensorDescription>,
        reshaped: impl Iterator<Item = (&'a TensorDescription, &'a TensorDescription, bool)>,
    ) {
        vectorization_default(vectorizations, handles_inputs, inputs, outputs, reshaped)
    }
}

fn vectorization_default<'a, R: JitRuntime>(
    vectorizations: &mut BTreeMap<TensorId, u8>,
    handles_inputs: impl Iterator<Item = &'a JitFusionHandle<R>>,
    inputs: impl Iterator<Item = &'a TensorDescription>,
    outputs: impl Iterator<Item = &'a TensorDescription>,
    reshaped: impl Iterator<Item = (&'a TensorDescription, &'a TensorDescription, bool)>,
) {
    enum Vect {
        Broadcated,
        Max(u8),
    }

    // The default version uses the last dimension as vectorization axis and assumes a
    // perpendicular contiguous line.
    let vectorization_input = |handle: &JitFusionHandle<R>, desc: &TensorDescription| {
        let rank = handle.strides.len();

        // Last dimension strides should be 1, otherwise vecX won't be contiguous.
        if handle.strides[rank - 1] != 1 {
            return Vect::Max(1);
        }
        let shape_axis = desc.shape[rank - 1];

        if shape_axis == 1 {
            return Vect::Broadcated;
        }

        for s in R::line_size_elem(&desc.dtype.into()) {
            // The last dimension should be a multiple of the vector size or broadcated.
            if shape_axis % s as usize == 0 {
                return Vect::Max(s);
            }
        }

        Vect::Max(1)
    };

    let vectorization_output = |desc: &TensorDescription| {
        let rank = desc.shape.len();

        for s in R::line_size_elem(&desc.dtype.into()) {
            // The last dimension should be a multiple of the vector size.
            if desc.shape[rank - 1] % s as usize == 0 {
                return Vect::Max(s);
            }
        }

        Vect::Max(1)
    };

    let vectorization_reshape =
        |reshaped: &TensorDescription, original: &TensorDescription, multi_reads: bool| {
            let reshape_axis = reshaped.shape[reshaped.shape.len() - 1];
            let shape_axis = original.shape[original.shape.len() - 1];

            if !multi_reads && reshape_axis == 1 {
                return Vect::Broadcated;
            }

            for s in R::line_size_elem(&reshaped.dtype.into()) {
                if !multi_reads {
                    // The last dimension should be a multiple of the vector size or broadcated.
                    if reshape_axis % s as usize == 0 {
                        return Vect::Max(s);
                    }
                } else {
                    // Since the original tensor must share the same vectorization factor as the
                    // reshaped tensor, they must have compatible shapes when both are access
                    // independently.
                    if reshape_axis % s as usize == 0 && shape_axis % s as usize == 0 {
                        return Vect::Max(s);
                    }
                }
            }

            Vect::Max(1)
        };

    let mut max_current = u8::MAX;

    for (handle, tensor) in handles_inputs.zip(inputs) {
        match vectorization_input(handle, tensor) {
            Vect::Broadcated => vectorizations.insert(tensor.id, 1),
            Vect::Max(val) => {
                max_current = Ord::min(val, max_current);
                vectorizations.insert(tensor.id, 0)
            }
        };
    }

    for tensor in outputs {
        match vectorization_output(tensor) {
            Vect::Broadcated => vectorizations.insert(tensor.id, 1),
            Vect::Max(val) => {
                max_current = Ord::min(val, max_current);
                vectorizations.insert(tensor.id, 0)
            }
        };
    }

    for (reshaped, original, multi_reads) in reshaped {
        match vectorization_reshape(reshaped, original, multi_reads) {
            Vect::Broadcated => {
                vectorizations.insert(original.id, 1);
                vectorizations.insert(reshaped.id, 1);
            }
            Vect::Max(val) => {
                vectorizations.insert(original.id, 0);
                vectorizations.insert(reshaped.id, 0);
                max_current = Ord::min(val, max_current);
            }
        }
    }

    for (_id, val) in vectorizations.iter_mut() {
        if *val == 0 {
            *val = max_current;
        }
    }
}
