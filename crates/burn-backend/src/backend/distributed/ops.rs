use crate::{
    Backend,
    tensor::{Device, FloatTensor},
};

use crate::distributed::{
    DistributedConfig, DistributedParams, ReduceOperation,
    all_reduce::all_reduce_inplace_centralized, close_distributed_sync_server,
    get_distributed_sync_client, start_distributed_sync_server,
};

/// Mutable reference to a float tensor.
#[derive(Clone)]
pub struct TensorRef<B: Backend>(pub *mut FloatTensor<B>);
unsafe impl<B> Sync for TensorRef<B> where B: Backend {}
unsafe impl<B> Send for TensorRef<B> where B: Backend {}

// TODO : Once the change from `TypeId` to `DeviceId` is made in `backend/communication/api.rs`,
// we can move the client operations out of the trait (and crate), which eliminates the need for the `communication` feature flag.

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

    /// In-place version of all_reduce.
    ///
    /// # Arguments
    ///
    /// * `tensors` - The tensors on which to perform all_reduce.
    /// * `op` - The [`ReduceOperation`].
    ///
    /// # Safety
    ///
    /// Ensure that the tensors are not accessed/modified when calling in-place operation.
    #[allow(unused)]
    unsafe fn all_reduce_in_place(tensors: Vec<TensorRef<Self>>, op: ReduceOperation) {
        unsafe { all_reduce_inplace_centralized(tensors, op) };
    }

    /// Sync the collective operations.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to sync.
    #[allow(unused)]
    fn sync_collective(device: &Self::Device) {
        // Default implementation executes collective operations synchronously, so nothing to do here.
    }

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
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
