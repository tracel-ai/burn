use cubecl::device::DeviceId;

use crate::{
    Backend,
    distributed::CollectiveTensor,
    tensor::{Device, FloatTensor},
};

use crate::distributed::{
    DistributedConfig, DistributedParams, ReduceOperation, close_distributed_sync_server,
    get_distributed_sync_client, start_distributed_sync_server,
};

/// Mutable reference to a float tensor.
#[derive(Clone)]
pub struct TensorRef<B: Backend>(pub *mut FloatTensor<B>);
unsafe impl<B> Sync for TensorRef<B> where B: Backend {}
unsafe impl<B> Send for TensorRef<B> where B: Backend {}

// TODO : The following functions should be moved in the `burn-autodiff` crate. The difficulty is in not discriminating between
// the dispatch backend and its inner dispatched backend when calling the communication server API. This implementation makes
// it easy by implementing `DisributedBackend` for `DispatchBackend`.
// The functions in question :
// * `start_communication_server`
// * `close_communication_server`
// * `register_sync_parameters`
// * `submit_sync_collective`
// * `submit_gradient_sync`

/// Operations on communication tensors.
pub trait DistributedBackend: Backend {
    /// Start the communication server used to orchestrate tensor syncing between devices.
    ///
    /// # Arguments
    ///
    /// * `devices` - The devices to orchestrate.
    fn start_communication_server(devices: &[Self::Device], config: DistributedConfig) {
        start_distributed_sync_server::<Self>(devices, config);
    }

    /// Close the communication server used to orchestrate syncing between devices.
    ///
    /// # Arguments
    ///
    /// * `device` - A device on the backend.
    fn close_communication_server(_device: &Self::Device) {
        close_distributed_sync_server::<Self>();
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
    fn register_sync_parameters(
        _device: &Self::Device,
        distributed_params: Vec<DistributedParams>,
    ) {
        if let Some(sync_client) = get_distributed_sync_client::<Self>() {
            sync_client.register_sync_parameters(distributed_params);
        };
    }

    /// Tell the gradient sync server that this device has submitted all its sync operations and is ready to be synchronized.
    ///
    /// # Arguments
    ///
    /// * `device` - The device on which to sync.
    fn submit_sync_collective(device: &Self::Device) {
        if let Some(sync_client) = get_distributed_sync_client::<Self>() {
            sync_client.submit_sync_collective(device.clone());
        };
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
    fn submit_gradient_sync(tensor: TensorRef<Self>, distributed_params: DistributedParams) {
        if let Some(sync_client) = get_distributed_sync_client::<Self>() {
            sync_client.submit_gradient_sync(tensor, distributed_params);
        };
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
        _tensor: FloatTensor<Self>,
        _op: ReduceOperation,
        _device_ids: Vec<DeviceId>,
    ) -> CollectiveTensor<Self> {
        unimplemented!()
    }

    /// Sync the collective operations.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to sync.
    fn sync_collective(_device: &Self::Device) {
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
    unsafe fn comm_device(tensor: &TensorRef<Self>) -> Device<Self> {
        unsafe { Self::float_device(&(*tensor.0)) }
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
    unsafe fn float_from_ref(tensor: &TensorRef<Self>) -> FloatTensor<Self> {
        unsafe { (*tensor.0).clone() }
    }
}
