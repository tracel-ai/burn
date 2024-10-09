use crate::fusion::on_write::kernel::fuse_on_write;
use crate::{fusion::JitFusionHandle, JitRuntime};
use burn_fusion::stream::Context;
use burn_tensor::repr::TensorDescription;
use cubecl::{calculate_cube_count_elemwise, client::ComputeClient, prelude::*, CubeDim};
use serde::{Deserialize, Serialize};

use crate::fusion::on_write::{
    ir::{Arg, ElemwiseConfig, ElemwisePrecision, GlobalArgs, GlobalArgsLaunch},
    trace::{FuseOnWriteTrace, TraceRunner},
};

#[derive(new)]
/// Fuse element wise operations into a single kernel.
pub struct ElemwiseOptimization<R: JitRuntime> {
    trace: FuseOnWriteTrace,
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    len: usize,
}

#[derive(Serialize, Deserialize)]
/// State for the [elemwise optimization](ElemwiseOptimization).
pub struct ElemwiseOptimizationState {
    trace: FuseOnWriteTrace,
    len: usize,
}

impl<R: JitRuntime> ElemwiseOptimization<R> {
    /// Execute the optimization.
    pub fn execute(&mut self, context: &mut Context<'_, JitFusionHandle<R>>) {
        self.trace
            .run::<R, Self>(&self.client, &self.device, context)
    }

    /// Number of element wise operations fused.
    pub fn num_ops_fused(&self) -> usize {
        self.len
    }

    /// Create an optimization from its [state](ElemwiseOptimizationState).
    pub fn from_state(device: &R::Device, state: ElemwiseOptimizationState) -> Self {
        Self {
            trace: state.trace,
            len: state.len,
            client: R::client(device),
            device: device.clone(),
        }
    }

    /// Convert the optimization to its [state](ElemwiseOptimizationState).
    pub fn to_state(&self) -> ElemwiseOptimizationState {
        ElemwiseOptimizationState {
            trace: self.trace.clone(),
            len: self.len,
        }
    }
}

impl<R: JitRuntime> TraceRunner<R> for ElemwiseOptimization<R> {
    fn run<'a>(
        client: &ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        config: ElemwiseConfig,
    ) {
        let arg = match config.ref_layout {
            Arg::Input(index, precision, _) => match precision {
                ElemwisePrecision::F32 => inputs.t_f32.values.get(index as usize),
                ElemwisePrecision::F16 => inputs.t_f16.values.get(index as usize),
                ElemwisePrecision::U32 => inputs.t_u32.values.get(index as usize),
                ElemwisePrecision::I32 => inputs.t_i32.values.get(index as usize),
                _ => panic!("Invalid value"),
            },
            Arg::Output(index, precision, _) => match precision {
                ElemwisePrecision::F32 => outputs.t_f32.values.get(index as usize),
                ElemwisePrecision::F16 => outputs.t_f16.values.get(index as usize),
                ElemwisePrecision::U32 => outputs.t_u32.values.get(index as usize),
                ElemwisePrecision::I32 => outputs.t_i32.values.get(index as usize),
                _ => panic!("Invalid value"),
            },
            _ => panic!("Invalid value"),
        };
        let (shape, vectorization) = match arg {
            Some(val) => match val {
                TensorArg::Handle {
                    handle,
                    vectorization_factor,
                } => (handle.shape, vectorization_factor),
                _ => panic!("Can't be an alias"),
            },
            None => panic!("Invalid argument"),
        };

        let total_elem = shape.iter().product::<usize>() / *vectorization as usize;
        let cube_dim = CubeDim::default();
        let cube_count = calculate_cube_count_elemwise(total_elem, cube_dim);

        unsafe {
            elemwise_fuse::launch_unchecked(client, cube_count, cube_dim, inputs, outputs, config);
        }
    }

    fn vectorization<'a>(
        handles_inputs: impl Iterator<Item = &'a JitFusionHandle<R>>,
        inputs: impl Iterator<Item = &'a TensorDescription>,
        outputs: impl Iterator<Item = &'a TensorDescription>,
    ) -> u8 {
        let factors = R::supported_line_sizes();

        let vectorization_input = |handle: &JitFusionHandle<R>, desc: &TensorDescription| {
            let rank = handle.strides.len();

            // Last dimension strides should be 1, otherwise vecX won't be contiguous.
            if handle.strides[rank - 1] != 1 {
                return 1;
            }

            for s in factors {
                // The last dimension should be a multiple of the vector size.
                if desc.shape[rank - 1] % *s as usize == 0 {
                    return *s;
                }
            }

            1
        };

        let vectorization_output = |desc: &TensorDescription| {
            let rank = desc.shape.len();

            for s in factors {
                // The last dimension should be a multiple of the vector size.
                if desc.shape[rank - 1] % *s as usize == 0 {
                    return *s;
                }
            }

            1
        };

        let mut output = u8::MAX;

        for (handle, tensor) in handles_inputs.zip(inputs) {
            output = u8::min(vectorization_input(handle, tensor), output);
        }

        for tensor in outputs {
            output = u8::min(vectorization_output(tensor), output);
        }

        output
    }
}

#[cube(launch_unchecked)]
fn elemwise_fuse(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    #[comptime] config: &ElemwiseConfig,
) {
    // We write no values for this fusion.
    let values = Registry::<Arg, Line<f32>>::new();
    let args = comptime![Sequence::<Arg>::new()];
    let pos = ABSOLUTE_POS;

    let length = match comptime![config.ref_layout] {
        Arg::Input(index, precision, _) => match comptime![precision] {
            ElemwisePrecision::F32 => inputs.t_f32.index(index).len(),
            ElemwisePrecision::F16 => inputs.t_f16.index(index).len(),
            ElemwisePrecision::U32 => inputs.t_u32.index(index).len(),
            ElemwisePrecision::I32 => inputs.t_i32.index(index).len(),
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        Arg::Output(index, precision, _) => match comptime![precision] {
            ElemwisePrecision::F32 => outputs.t_f32.index(index).len(),
            ElemwisePrecision::F16 => outputs.t_f16.index(index).len(),
            ElemwisePrecision::U32 => outputs.t_u32.index(index).len(),
            ElemwisePrecision::I32 => outputs.t_i32.index(index).len(),
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        _ => comptime![panic!("Invalid ref layout.")],
    };

    if pos < length {
        fuse_on_write::<f32>(inputs, outputs, pos, values, args, config)
    }
}
