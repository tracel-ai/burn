use alloc::{string::String, vec::Vec};
use burn_ir::TensorIr;
use burn_tensor::{backend::DeviceOps, DType, Element};

use crate::{get_client, MultiBackendBridge, RouterTensor, RunnerClient};

/// Type alias for `<Br as MultiBackendBridge>::TensorHandle`.
pub type TensorHandle<Br> = <Br as MultiBackendBridge>::TensorHandle;

/// Defines the connection channel and operations for a setup with multiple backend runner clients.
pub trait RunnerChannel: Clone + Send + Sync + 'static + Sized {
    /// Device type.
    type Device: DeviceOps;
    /// A bridge that can transfer tensors between multiple backends.
    type Bridge: MultiBackendBridge<Device = Self::Device>;
    /// Client type.
    type Client: RunnerClient<Device = Self::Device>;
    /// Float element type.
    type FloatElem: Element;
    /// Int element type.
    type IntElem: Element;
    /// Bool element type.
    type BoolElem: Element;

    /// Name of the channel.
    fn name() -> String;

    /// Initialize a new client for the given device.
    fn init_client(device: &Self::Device) -> Self::Client;

    /// Get the tensor handle corresponding to the [tensor representation](TensorIr).
    fn get_tensor_handle(tensor: &TensorIr, client: &Self::Client) -> TensorHandle<Self::Bridge>;

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
        let desc = tensor.into_ir();
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
