use crate::fusion::on_write::kernel::fuse_on_write;
use crate::{fusion::JitFusionHandle, JitRuntime};
use burn_fusion::stream::Context;
use cubecl::{calculate_cube_count_elemwise, client::ComputeClient, prelude::*, CubeDim};
use serde::{Deserialize, Serialize};

use crate::fusion::on_write::{
    ir::{Arg, FusionArgs, FusionArgsLaunch, FusionConfig, OpPrecision},
    trace::{FuseOnWriteTrace, Launch},
};

#[derive(new)]
pub struct ElemwiseKernel<R: JitRuntime> {
    trace: FuseOnWriteTrace,
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    len: usize,
}

#[derive(Serialize, Deserialize)]
pub struct ElemwiseKernelState {
    trace: FuseOnWriteTrace,
    len: usize,
}

impl<R: JitRuntime> ElemwiseKernel<R> {
    pub fn execute(&mut self, context: &mut Context<'_, JitFusionHandle<R>>) {
        let vectorization = 4;

        self.trace
            .run::<R, Self>(&self.client, &self.device, vectorization, context)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn from_state(device: &R::Device, state: ElemwiseKernelState) -> Self {
        Self {
            trace: state.trace,
            len: state.len,
            client: R::client(&device),
            device: device.clone(),
        }
    }
    pub fn to_state(&self) -> ElemwiseKernelState {
        ElemwiseKernelState {
            trace: self.trace.clone(),
            len: self.len.clone(),
        }
    }
}

impl<R: JitRuntime> Launch<R> for ElemwiseKernel<R> {
    fn run<'a>(
        client: &ComputeClient<R::Server, R::Channel>,
        inputs: FusionArgsLaunch<'a, R>,
        outputs: FusionArgsLaunch<'a, R>,
        config: FusionConfig,
    ) {
        let arg = config.ref_layout.arg;
        let arg = match arg {
            Arg::Input(index, precision) => match precision {
                OpPrecision::F32 => inputs.t_f32.values.get(index as usize),
                OpPrecision::F16 => inputs.t_f16.values.get(index as usize),
                _ => panic!("Invalid value"),
            },
            Arg::Output(index, precision) => match precision {
                OpPrecision::F32 => outputs.t_f32.values.get(index as usize),
                OpPrecision::F16 => outputs.t_f16.values.get(index as usize),
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
            elemwise_fuse::launch_unchecked(client, cube_count, cube_dim, inputs, outputs, config)
        }
    }
}

#[cube(launch_unchecked)]
pub fn elemwise_fuse(
    inputs: &FusionArgs,
    outputs: &mut FusionArgs,
    #[comptime] config: &FusionConfig,
) {
    fuse_on_write::<f32>(inputs, outputs, ABSOLUTE_POS, Line::empty(1), None, config)
}
