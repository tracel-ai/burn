use crate::optim::FusedOperation;
use crate::{
    CubeFusionHandle, FallbackOperation,
    engine::{
        codegen::{
            DynSize,
            io::ref_len,
            ir::{
                FuseArg, FuseBlockConfig, GlobalArgs, GlobalArgsLaunch, RefLayout,
                multi_block_variables_init,
            },
            kernel::{fuse_on_write, init_locals},
        },
        launch::{
            FuseTraceLauncher,
            runner::{TraceRunner, Vectorization},
        },
        trace::FuseTrace,
    },
};
use burn_fusion::stream::Context;
use cubecl::{CubeDim, calculate_cube_count_elemwise, client::Client, prelude::*};
use serde::{Deserialize, Serialize};

#[derive(new)]
/// Fuse element wise operations into a single kernel.
pub struct ElemwiseOptimization {
    pub(crate) trace: FuseTrace,
    client: Client,
    device: cubecl::Device,
    len: usize,
}

#[derive(Serialize, Deserialize, Debug)]
/// State for the [elemwise optimization](ElemwiseOptimization).
pub struct ElemwiseOptimizationState {
    trace: FuseTrace,
    len: usize,
}

impl ElemwiseOptimization {
    /// Execute the optimization.
    pub fn execute(&self, context: &mut Context<CubeFusionHandle>) {
        let launcher = FuseTraceLauncher::new(&self.trace, &ElemwiseRunner);

        match launcher.launch(&self.client, &self.device, context) {
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
    pub fn from_state(device: &cubecl::Device, state: ElemwiseOptimizationState) -> Self {
        Self {
            trace: state.trace,
            len: state.len,
            client: device.client(),
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

impl Vectorization for ElemwiseRunner {}
impl TraceRunner for ElemwiseRunner {
    type Error = LaunchError; // No error possible

    fn run<'a>(
        &'a self,
        client: &'a Client,
        inputs: GlobalArgsLaunch,
        outputs: GlobalArgsLaunch,
        configs: &[FuseBlockConfig],
    ) -> Result<(), Self::Error> {
        let config = &configs[0];
        let shape = match &config.ref_layout {
            RefLayout::Concrete(arg) => match arg {
                FuseArg::Input(..) => inputs.shape_ref(&config.ref_layout, config.rank),
                FuseArg::Output(..) => outputs.shape_ref(&config.ref_layout, config.rank),
                _ => panic!("Invalid concreate ref layout"),
            },
            RefLayout::Virtual(_) => inputs.shape_ref(&config.ref_layout, config.rank),
        };
        let working_units = shape.iter().product::<usize>() / config.width;
        let cube_dim = CubeDim::new(client, working_units);
        let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
        let address_type = inputs
            .required_address_type()
            .max(outputs.required_address_type());

        unsafe {
            elemwise_fuse::launch_unchecked(
                client,
                cube_count,
                cube_dim,
                address_type,
                inputs,
                outputs,
                config.clone(),
            );
        };

        Ok(())
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn elemwise_fuse(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    #[comptime] config: FuseBlockConfig,
) {
    // We write no values for this fusion.
    let values = Registry::<FuseArg, Vector<f32, DynSize>>::new();
    let args = comptime![Vec::<FuseArg>::new()];
    let pos = ABSOLUTE_POS;

    multi_block_variables_init(&config, &mut outputs.variables);

    let mut locals = init_locals(inputs, outputs, &config);
    let length = ref_len(inputs, &*outputs, &locals, &config);

    if pos < length {
        fuse_on_write::<f32, DynSize>(inputs, outputs, &mut locals, pos, values, args, &config)
    }
}

/// Name of the element-wise fusion optimization.
pub const NAME: &str = "ElementWise";

impl FusedOperation for ElemwiseOptimization {
    fn max_relative_shape_id(&self) -> Option<usize> {
        self.trace.max_relative_shape_id()
    }

    const NAME: &'static str = self::NAME;
    type State = ElemwiseOptimizationState;

    fn num_ops_fused(&self) -> usize {
        Self::num_ops_fused(self)
    }

    fn run(
        &mut self,
        context: &mut Context<CubeFusionHandle>,
        _fallback: &dyn Fn(usize) -> Box<dyn FallbackOperation>,
    ) {
        Self::execute(self, context)
    }

    fn to_state(&self) -> Self::State {
        Self::to_state(self)
    }

    fn from_state(device: &cubecl::Device, state: Self::State) -> Self {
        Self::from_state(device, state)
    }
}
