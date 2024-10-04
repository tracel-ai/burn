use crate::{
    backend::DeviceOps,
    repr::{ReprBackend, TensorDescription},
    router::{get_client, MultiBackendBridge, RouterTensor, RunnerClient},
    DType,
};

// Defines associated types config for a setup with multiple backend runners.
pub trait RunnerChannel: Clone + Send + Sync + 'static + Sized {
    type Device: DeviceOps;
    type Bridge: MultiBackendBridge<Device = Self::Device>;
    type Client: RunnerClient<Device = Self::Device>;

    /// Initialize a new client for the given device.
    fn init_client(device: &Self::Device) -> Self::Client;

    fn get_float_tensor(
        tensor: &TensorDescription,
        client: Self::Client,
    ) -> <Self::Bridge as MultiBackendBridge>::TensorType;

    fn get_int_tensor(
        tensor: &TensorDescription,
        client: Self::Client,
    ) -> <Self::Bridge as MultiBackendBridge>::TensorType;

    fn get_bool_tensor(
        tensor: &TensorDescription,
        client: Self::Client,
    ) -> <Self::Bridge as MultiBackendBridge>::TensorType;

    fn get_quantized_tensor(
        tensor: &TensorDescription,
        client: Self::Client,
    ) -> <Self::Bridge as MultiBackendBridge>::TensorType;

    /// Create a tensor with the given handle and shape.
    fn register_tensor(
        client: Self::Client,
        handle: <Self::Bridge as MultiBackendBridge>::TensorType,
        shape: Vec<usize>,
        dtype: DType,
    ) -> RouterTensor<Self::Client>;

    /// Change the tensor to a different backend runner.
    fn change_backend(
        tensor: RouterTensor<Self::Client>,
        device: &Self::Device, // target device
    ) -> RouterTensor<Self::Client> {
        // To go from TensorDescription -> TensorPrimitive: <HandleContainer>.get_float_tensor(desc)
        // The handle container is in the runner, which implements RunnerClient
        // So we must go from RouterTensor -> TensorDescription -> TensorPrimitive (TensorType)
        let original_client = tensor.client.clone();
        let desc = tensor.into_description();
        let target_client = get_client::<Self>(device);
        let tensor = Self::get_float_tensor(&desc, original_client);
        // let target_tensor = Self::Bridge::to_backend(tensor, device);
        let handle = Self::Bridge::change_backend_float(tensor, desc.shape.clone().into(), device);

        Self::register_tensor(target_client, handle, desc.shape, desc.dtype)
        // TODO: remove tensor handle from other client?
    }
}
