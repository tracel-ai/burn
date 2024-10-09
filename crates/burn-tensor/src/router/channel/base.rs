use crate::{
    backend::DeviceOps,
    repr::TensorDescription,
    router::{get_client, MultiBackendBridge, RouterTensor, RunnerClient},
    DType,
};

/// Type alias for `<Br as MultiBackendBridge>::TensorHandle`.
pub type TensorHandle<Br> = <Br as MultiBackendBridge>::TensorHandle;

// Defines associated types config for a setup with multiple backend runners.
// TODO: most of the stuff should go in the client no?
// We would then have something like a DirectClient or LocalClient instead of DirectChannel
// type MyBackend = BackendRouter<DirectClient<(Cuda, NdArray, Wgpu), ByteBridge<(Cuda, NdArray, Wgpu)>>>
// and in the future perhaps HttpClient
pub trait RunnerChannel: Clone + Send + Sync + 'static + Sized {
    type Device: DeviceOps;
    type Bridge: MultiBackendBridge<Device = Self::Device>;
    type Client: RunnerClient<Device = Self::Device>;

    /// Initialize a new client for the given device.
    fn init_client(device: &Self::Device) -> Self::Client;

    /// Get the tensor handle corresponding to the [tensor description](TensorDescription).
    fn get_tensor_handle(
        tensor: &TensorDescription,
        client: &Self::Client,
    ) -> TensorHandle<Self::Bridge>;

    // TODO: get quantized tensor handle from QuantizedTensorDescription?

    /// Create a tensor with the given handle and shape.
    fn register_tensor(
        client: &Self::Client,
        handle: TensorHandle<Self::Bridge>,
        shape: Vec<usize>,
        dtype: DType,
    ) -> RouterTensor<Self::Client>;

    /// Change the tensor to a different client backend.
    fn change_client_backend(
        tensor: RouterTensor<Self::Client>,
        device: &Self::Device, // target device
    ) -> RouterTensor<Self::Client> {
        // Get tensor handle from current client
        let original_client = tensor.client.clone();
        let desc = tensor.into_description();
        let mut handle = Self::get_tensor_handle(&desc, &original_client);

        if desc.dtype.is_float() {
            handle = Self::Bridge::change_backend_float(handle, desc.shape.clone().into(), device);
        } else if desc.dtype.is_int() {
            handle = Self::Bridge::change_backend_int(handle, desc.shape.clone().into(), device);
        } else if desc.dtype.is_bool() {
            handle = Self::Bridge::change_backend_bool(handle, desc.shape.clone().into(), device);
        } else {
            unimplemented!()
        }

        // Register tensor handle on target client
        let target_client = get_client::<Self>(device);
        Self::register_tensor(&target_client, handle, desc.shape, desc.dtype)
    }
}
