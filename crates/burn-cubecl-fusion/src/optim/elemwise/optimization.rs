use crate::{
    CubeFusionHandle,
    engine::{
        io::ref_len,
        ir::{FuseArg, FuseBlockConfig, GlobalArgs, GlobalArgsLaunch, RefLayout},
        kernel::{fuse_on_write, init_locals},
        trace::{FuseTrace, TraceRunner, Vectorization},
    },
};
use burn_fusion::stream::Context;
use cubecl::{CubeDim, calculate_cube_count_elemwise, client::ComputeClient, prelude::*};
use serde::{Deserialize, Serialize};

#[derive(new)]
/// Fuse element wise operations into a single kernel.
pub struct ElemwiseOptimization<R: Runtime> {
    trace: FuseTrace,
    client: ComputeClient<R::Server>,
    device: R::Device,
    len: usize,
}

#[derive(Serialize, Deserialize, Debug)]
/// State for the [elemwise optimization](ElemwiseOptimization).
pub struct ElemwiseOptimizationState {
    trace: FuseTrace,
    len: usize,
}

impl<R: Runtime> ElemwiseOptimization<R> {
    /// Execute the optimization.
    pub fn execute<BT: CubeElement>(&mut self, context: &mut Context<'_, CubeFusionHandle<R>>) {
        match self.trace.run::<R, BT, ElemwiseRunner>(
            &self.client,
            &self.device,
            context,
            &ElemwiseRunner,
        ) {
            Ok(_) => (),
            Err(err) => {
                panic!("{err:?} - {:?}", self.trace);
            }
        }
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

impl<R: Runtime> Vectorization<R> for ElemwiseRunner {}
impl<R: Runtime> TraceRunner<R> for ElemwiseRunner {
    type Error = (); // No error possible

    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R::Server>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        configs: &[FuseBlockConfig],
    ) -> Result<(), Self::Error> {
        let config = &configs[0];
        let shape = match &config.ref_layout {
            RefLayout::Concrete(arg) => match arg {
                FuseArg::Input(..) => inputs.shape_ref(&config.ref_layout, config.rank as usize),
                FuseArg::Output(..) => outputs.shape_ref(&config.ref_layout, config.rank as usize),
                _ => panic!("Invalid concreate ref layout"),
            },
            RefLayout::Virtual(_) => inputs.shape_ref(&config.ref_layout, config.rank as usize),
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
    #[comptime] config: &FuseBlockConfig,
) {
    // We write no values for this fusion.
    let values = Registry::<FuseArg, Line<f32>>::new();
    let args = comptime![Sequence::<FuseArg>::new()];
    let pos = ABSOLUTE_POS;

    let mut locals = init_locals(inputs, outputs, config);
    let length = ref_len(inputs, outputs, &locals, config);

    if pos < length {
        fuse_on_write::<f32>(inputs, outputs, &mut locals, pos, values, args, config)
    }
}
