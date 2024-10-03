use super::ir::FusionArgs;
use crate::fusion::elemwise2::kernel::fuse_on_write;
use crate::{fusion::JitFusionHandle, JitRuntime};
use burn_fusion::stream::Context;
use cubecl::{calculate_cube_count_elemwise, client::ComputeClient, prelude::*, CubeDim};

use super::{
    ir::{Arg, FusionArgsLaunch, FusionConfig, OpPrecision},
    trace::{Launch, Tracel2},
};

#[derive(new)]
pub struct ElemwiseKernel<R: JitRuntime> {
    trace: Tracel2,
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    len: usize,
}

impl<R: JitRuntime> ElemwiseKernel<R> {
    pub fn execute(&mut self, context: &mut Context<'_, JitFusionHandle<R>>) {
        let vectorization = 1;

        println!("Run");
        self.trace
            .run::<R, Self>(&self.client, &self.device, vectorization, context)
    }

    pub fn len(&self) -> usize {
        self.len
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
        let shape = match arg {
            Some(val) => match val {
                TensorArg::Handle { handle, .. } => handle.shape,
                _ => panic!("Can't be an alias"),
            },
            None => panic!("Invalud argument"),
        };

        let total_elem = shape.iter().product();
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
