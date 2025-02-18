use crate::on_write::ir::GlobalArgs;
use crate::on_write::{io::global_length, kernel::fuse_on_write};
use crate::CubeFusionHandle;
use burn_fusion::stream::Context;
use cubecl::{calculate_cube_count_elemwise, client::ComputeClient, prelude::*, CubeDim};
use serde::{Deserialize, Serialize};

use crate::on_write::{
    ir::{Arg, ElemwiseConfig, GlobalArgsLaunch},
    trace::{FuseOnWriteTrace, TraceRunner},
};

#[derive(new)]
/// Fuse element wise operations into a single kernel.
pub struct ElemwiseOptimization<R: Runtime> {
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

impl<R: Runtime> ElemwiseOptimization<R> {
    /// Execute the optimization.
    pub fn execute<BT: CubeElement>(&mut self, context: &mut Context<'_, CubeFusionHandle<R>>) {
        self.trace
            .run::<R, BT, ElemwiseRunner>(&self.client, &self.device, context, &ElemwiseRunner)
            .unwrap();
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

pub struct ElemwiseRunner;

impl<R: Runtime> TraceRunner<R> for ElemwiseRunner {
    type Error = (); // No error possible

    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        config: &'a ElemwiseConfig,
    ) -> Result<(), Self::Error> {
        let arg = match config.ref_layout {
            Arg::Input(index, _, _) => inputs.tensors.values.get(index as usize),
            Arg::Output(index, _, _) => outputs.tensors.values.get(index as usize),
            _ => panic!("Invalid value"),
        };
        let shape = match arg {
            Some(val) => match &val.tensor {
                TensorArg::Handle { handle, .. } => handle.shape,
                TensorArg::Alias { .. } => panic!("Can't be an alias, got {val:?}"),
            },
            None => panic!("Invalid argument"),
        };
        let total_elem = shape.iter().product::<usize>() / config.width as usize;
        let cube_dim = CubeDim::default();
        let cube_count = calculate_cube_count_elemwise(total_elem, cube_dim);

        unsafe {
            elemwise_fuse::launch_unchecked(
                client,
                cube_count,
                cube_dim,
                inputs,
                outputs,
                config.clone(),
            );
        };

        Ok(())
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

    let length = match comptime![config.ref_layout.clone()] {
        Arg::Input(index, _, _) => global_length(inputs, index),
        Arg::Output(index, _, _) => global_length(outputs, index),
        _ => comptime![panic!("Invalid ref layout.")],
    };

    if pos < length {
        fuse_on_write::<f32>(inputs, outputs, pos, values, args, config)
    }
}
