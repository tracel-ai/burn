use crate::{
    backend::DeviceOps,
    router::{MultiBackendBridge, RunnerClient},
};

// Defines associated types config for a setup with multiple backend runners.
pub trait RunnerChannel: Clone + Send + Sync + 'static + Sized {
    type Device: DeviceOps;
    type Bridge: MultiBackendBridge<Device = Self::Device>;
    type Client: RunnerClient;

    /// Initialize a new client for the given device.
    fn init_client(device: &Self::Device) -> Self::Client;

    /// Change the tensor to a different runner.
    fn change_runner(
        self,
        tensor: <Self::Bridge as MultiBackendBridge>::TensorType,
        device: &Self::Device, // target device
    ) -> <Self::Bridge as MultiBackendBridge>::TensorType {
        Self::Bridge::to_backend(tensor, device)
    }
}
