use crate::{
    CubeFusionHandle, FallbackOperation,
    engine::{launch::FuseTraceLauncher, trace::FuseTrace},
    optim::elemwise::ElemwiseRunner,
};
use burn_fusion::stream::Context;
use cubecl::{client::ComputeClient, prelude::*};
use serde::{Deserialize, Serialize};

#[derive(new)]
/// Fuse element wise operations into a single kernel.
pub struct NHWCRelayoutOptimization<R: Runtime> {
    pub(crate) trace: FuseTrace,
    client: ComputeClient<R>,
    device: R::Device,
    len: usize,
}

#[derive(Serialize, Deserialize, Debug)]
/// State for the [elemwise optimization](RelayoutOptimization).
pub struct RelayoutOptimizationState {
    trace: FuseTrace,
    len: usize,
}

impl<R: Runtime> NHWCRelayoutOptimization<R> {
    pub fn execute(
        &self,
        context: &mut Context<CubeFusionHandle<R>>,
        fallback: impl FnOnce(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        let launcher_elemwise = FuseTraceLauncher::new(&self.trace, &ElemwiseRunner);

        match launcher_elemwise.launch(&self.client, &self.device, context) {
            Ok(_) => (),
            Err(err) => {
                panic!("{err:?} - {:?}", self.trace);
            }
        };
        fallback(self.len - 1).run(context);
    }

    /// Number of element wise operations fused.
    pub fn num_ops_fused(&self) -> usize {
        self.len
    }

    /// Create an optimization from its [state](RelayoutOptimizationState).
    pub fn from_state(device: &R::Device, state: RelayoutOptimizationState) -> Self {
        Self {
            trace: state.trace,
            len: state.len,
            client: R::client(device),
            device: device.clone(),
        }
    }

    /// Convert the optimization to its [state](RelayoutOptimizationState).
    pub fn to_state(&self) -> RelayoutOptimizationState {
        RelayoutOptimizationState {
            trace: self.trace.clone(),
            len: self.len,
        }
    }
}
