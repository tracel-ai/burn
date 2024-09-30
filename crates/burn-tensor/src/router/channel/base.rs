use crate::{
    backend::DeviceOps,
    router::{MultiBackendBridge, RunnerClient},
};

// Defines associated types config for a setup with multiple backend runners.
pub trait RunnerChannel: Clone + Send + Sync + 'static + Sized {
    type Device: DeviceOps;
    type Bridge: MultiBackendBridge;
    type Client: RunnerClient;

    /// Initialize a new client for the given device.
    fn init_client(device: &Self::Device) -> Self::Client;

    /// Change the tensor to a different runner.
    fn change_runner(
        self,
        tensor: <Self::Bridge as MultiBackendBridge>::TensorType,
    ) -> <Self::Bridge as MultiBackendBridge>::TensorType {
        // TODO: specify device?
        Self::Bridge::to_backend(&self, tensor)
    }
}
