use alloc::vec::Vec;

use crate::{
    Backend, DeviceId,
    distributed::CollectiveTensor,
    tensor::{Device, FloatTensor},
};

use crate::distributed::{DistributedConfig, DistributedParams, ReduceOperation};

#[cfg(feature = "std")]
use crate::distributed::{
    close_distributed_sync_server, get_distributed_sync_client, start_distributed_sync_server,
};

/// Mutable reference to a float tensor.
#[derive(Clone)]
pub struct TensorRef<B: Backend>(pub *mut FloatTensor<B>);
unsafe impl<B> Sync for TensorRef<B> where B: Backend {}
unsafe impl<B> Send for TensorRef<B> where B: Backend {}

// TODO : The following functions should be moved in the `burn-autodiff` crate. The difficulty is in not discriminating between
// the dispatch backend and its inner dispatched backend when calling the communication server API. This implementation makes
// it easy by implementing `DistributedOps` for `DispatchBackend`.
// The functions in question :
// * `start_communication_server`
// * `close_communication_server`
// * `register_sync_parameters`
// * `submit_sync_collective`
// * `submit_gradient_sync`

/// Operations on communication tensors.
pub trait DistributedOps<B: Backend> {
    /// Start the communication server used to orchestrate tensor syncing between devices.
    ///
    /// # Arguments
    ///
    /// * `devices` - The devices to orchestrate.
    fn start_communication_server(devices: &[B::Device], config: DistributedConfig) {
        #[cfg(feature = "std")]
        start_distributed_sync_server::<B>(devices, config);
        #[cfg(not(feature = "std"))]
        let _ = (devices, config);
    }

    /// Close the communication server used to orchestrate syncing between devices.
    ///
    /// # Arguments
    ///
    /// * `device` - A device on the backend.
    fn close_communication_server(_device: &B::Device) {
        #[cfg(feature = "std")]
        close_distributed_sync_server::<B>();
    }

    /// Register the parameters that will require gradient synchronization for the upcoming backward pass.
    ///
    /// This must be called before the backward pass on each device so the gradient sync server
    /// can coordinate collective operations across all devices.
    ///
    /// # Arguments
    ///
    /// * `device` - The device calling the initialization.
    /// * `distributed_params` - A list of [`DistributedParams`] of the tensors to sync.
    fn register_sync_parameters(_device: &B::Device, distributed_params: Vec<DistributedParams>) {
        #[cfg(feature = "std")]
        if let Some(sync_client) = get_distributed_sync_client::<B>() {
            sync_client.register_sync_parameters(distributed_params);
        };
        #[cfg(not(feature = "std"))]
        let _ = distributed_params;
    }

    /// Tell the gradient sync server that this device has submitted all its sync operations and is ready to be synchronized.
    ///
    /// # Arguments
    ///
    /// * `device` - The device on which to sync.
    fn submit_sync_collective(device: &B::Device) {
        #[cfg(feature = "std")]
        if let Some(sync_client) = get_distributed_sync_client::<B>() {
            sync_client.submit_sync_collective(device.clone());
        };
        #[cfg(not(feature = "std"))]
        let _ = device;
    }

    /// Submit a gradient tensor for synchronization across all devices.
    ///
    /// The gradient is sent to the gradient sync server, which will launch the all-reduce
    /// operation once all devices have submitted their gradient for this parameter.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to synchronize.
    /// * `distributed_params` - The [`DistributedParams`] for the parameter.
    fn submit_gradient_sync(tensor: TensorRef<B>, distributed_params: DistributedParams) {
        #[cfg(feature = "std")]
        if let Some(sync_client) = get_distributed_sync_client::<B>() {
            sync_client.submit_gradient_sync(tensor, distributed_params);
        };
        #[cfg(not(feature = "std"))]
        let _ = (tensor, distributed_params);
    }

    /// all_reduce operation.
    ///
    /// # Arguments
    ///
    /// * `tensors` - The tensors on which to perform all_reduce.
    /// * `op` - The [`ReduceOperation`].
    ///
    /// # Returns
    ///
    /// The corresponding [CollectiveTensor].
    fn all_reduce(
        _tensor: FloatTensor<B>,
        _op: ReduceOperation,
        _device_ids: Vec<DeviceId>,
    ) -> CollectiveTensor<B> {
        unimplemented!()
    }

    /// Sync the collective operations.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to sync.
    fn sync_collective(_device: &B::Device) {
        unimplemented!()
    }

    /// Get the device of the tensor reference.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor reference.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    ///
    /// # Safety
    ///
    /// Ensure that the tensors are not accessed/modified when calling.
    unsafe fn comm_device(tensor: &TensorRef<B>) -> Device<B> {
        unsafe { B::float_device(&(*tensor.0)) }
    }

    /// Returns a clone of the float tensor from the tensor reference.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing a copy of the data of the given tensor.
    ///
    /// # Safety
    ///
    /// Ensure that the tensors are not accessed/modified when calling.
    unsafe fn float_from_ref(tensor: &TensorRef<B>) -> FloatTensor<B> {
        unsafe { (*tensor.0).clone() }
    }
}
