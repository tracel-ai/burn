use core::marker::PhantomData;

use crate::{
    repr::ReprBackend,
    router::{Handle2, MultiBackendBridge, MultiDevice2, MultiRunnerClient2, Runner},
};

use super::RunnerChannel;

pub struct DirectChannel<Backends, Bridge> {
    backends: PhantomData<Backends>,
    bridge: PhantomData<Bridge>,
}

impl<Backends, Bridge> Clone for DirectChannel<Backends, Bridge> {
    fn clone(&self) -> Self {
        Self {
            backends: self.backends.clone(),
            bridge: self.bridge.clone(),
        }
    }
}

impl<B1: ReprBackend, B2: ReprBackend, Br: MultiBackendBridge<TensorType = Handle2<B1, B2>>>
    RunnerChannel for DirectChannel<(B1, B2), Br>
{
    type Device = MultiDevice2<B1, B2>;

    type Bridge = Br;

    type Client = MultiRunnerClient2<B1, B2>;

    fn init_client(device: &Self::Device) -> Self::Client {
        match device {
            MultiDevice2::Device1(device) => {
                MultiRunnerClient2::RunnerClient1(Runner::new(device.clone()))
            }
            MultiDevice2::Device2(device) => {
                MultiRunnerClient2::RunnerClient2(Runner::new(device.clone()))
            }
        }
    }
}
