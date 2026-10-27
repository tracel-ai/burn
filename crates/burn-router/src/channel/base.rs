use alloc::{string::String, vec::Vec};
use burn_backend::{
    DType, Shape, backend::DeviceOps, distributed::ReduceOperation, ops::EmulatedCollectives,
};
use burn_ir::TensorIr;

use crate::{BackendRouter, MultiBackendBridge, RouterClient, RouterTensor, get_client};

/// Type alias for `<Br as MultiBackendBridge>::TensorHandle`.
pub type TensorHandle<Br> = <Br as MultiBackendBridge>::TensorHandle;

/// Defines the connection channel and operations for a setup with multiple backend router clients.
pub trait RouterChannel: Clone + Send + Sync + 'static + Sized {
    /// Device type.
    type Device: DeviceOps;
    /// A bridge that can transfer tensors between multiple backends.
    type Bridge: MultiBackendBridge<Device = Self::Device>;
    /// Client type.
    type Client: RouterClient<Device = Self::Device>;

    /// Name of the channel.
    fn name(device: &Self::Device) -> String;

    /// Initialize a new unscoped client for the given device.
    ///
    /// [`get_client`] calls this on a cache miss and retains the result in the global client
    /// locator. Channels whose client lifetime is externally scoped can instead construct their
    /// client directly and install it with [`crate::register_scoped_client`]; that path deliberately
    /// bypasses this method.
    fn init_client(device: &Self::Device) -> Self::Client;

    /// Get the tensor handle corresponding to the [tensor representation](TensorIr).
    fn get_tensor_handle(tensor: &TensorIr, client: &Self::Client) -> TensorHandle<Self::Bridge>;

    /// Create a tensor with the given handle and shape.
    fn register_tensor(
        client: &Self::Client,
        handle: TensorHandle<Self::Bridge>,
        shape: Shape,
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
            handle = Self::Bridge::change_backend_float(handle, desc.shape.clone(), device);
        } else if desc.dtype.is_int() || desc.dtype.is_uint() {
            handle = Self::Bridge::change_backend_int(handle, desc.shape.clone(), device);
        } else if desc.dtype.is_bool() {
            handle = Self::Bridge::change_backend_bool(handle, desc.shape.clone(), device);
        } else {
            unimplemented!("moving a {:?} tensor to another router client", desc.dtype)
        }

        // Register tensor handle on target client
        let target_client = get_client::<Self>(device);
        Self::register_tensor(&target_client, handle, desc.shape, desc.dtype)
    }

    /// Every member's [all-reduce](burn_backend::ops::CollectiveOps::float_all_reduce), emulated
    /// with transfers unless the channel runs it where the members live.
    fn float_all_reduce(
        tensors: Vec<RouterTensor<Self::Client>>,
        op: ReduceOperation,
    ) -> Vec<RouterTensor<Self::Client>> {
        EmulatedCollectives::<BackendRouter<Self>>::float_all_reduce(tensors, op)
    }
}
