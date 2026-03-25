use alloc::vec::Vec;

use crate::{
    Backend, DistributedConfig, DistributedParams, ReduceOperation,
    all_reduce::all_reduce_inplace_centralized,
    tensor::{Device, FloatTensor},
};

#[cfg(feature = "communication")]
use crate::{
    close_distributed_sync_server, get_distributed_sync_client, start_distributed_sync_server,
};

/// Mutable reference to a float tensor.
#[derive(Clone)]
pub struct TensorRef<B: Backend>(pub *mut FloatTensor<B>);
unsafe impl<B> Sync for TensorRef<B> where B: Backend {}
unsafe impl<B> Send for TensorRef<B> where B: Backend {}

// TODO : Once the change from `TypeId` to `DeviceId` is made in `backend/communication/api.rs`,
// we can move the client operations out of the trait (and crate), which eliminates the need for the `communication` feature flag.

/// Operations on communication tensors.
pub trait CommunicationTensorOps<B: Backend> {
    /// Start the communication server used to orchestrate tensor syncing between devices.
    ///
    /// # Arguments
    ///
    /// * `devices` - The devices to orchestrate.
    fn start_communication_server(devices: Vec<B::Device>, config: DistributedConfig) {
        #[cfg(feature = "communication")]
        start_distributed_sync_server::<B>(devices, config);
        #[cfg(not(feature = "communication"))]
        unimplemented!()
    }

    /// Close the communication server used to orchestrate syncing between devices.
    ///
    /// # Arguments
    ///
    /// * `device` - A device on the backend.
    fn close_communication_server(_device: &B::Device) {
        #[cfg(feature = "communication")]
        close_distributed_sync_server::<B>();
        #[cfg(not(feature = "communication"))]
        unimplemented!()
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
        #[cfg(feature = "communication")]
        if let Some(sync_client) = get_distributed_sync_client::<B>() {
            sync_client.register_sync_parameters(distributed_params);
        };
        #[cfg(not(feature = "communication"))]
        unimplemented!()
    }

    /// Tell the gradient sync server that this device has submitted all its sync operations and is ready to be synchronized.
    ///
    /// # Arguments
    ///
    /// * `device` - The device on which to sync.
    fn submit_sync_collective(device: &B::Device) {
        #[cfg(feature = "communication")]
        if let Some(sync_client) = get_distributed_sync_client::<B>() {
            sync_client.submit_sync_collective(device.clone());
        };
        #[cfg(not(feature = "communication"))]
        unimplemented!()
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
        #[cfg(feature = "communication")]
        if let Some(sync_client) = get_distributed_sync_client::<B>() {
            sync_client.submit_gradient_sync(tensor, distributed_params);
        };
        #[cfg(not(feature = "communication"))]
        unimplemented!()
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
    unsafe fn all_reduce_in_place(tensors: Vec<TensorRef<B>>, op: ReduceOperation) {
        println!("Fallback op.");
        unsafe { all_reduce_inplace_centralized(tensors, op) };
    }

    /// Sync the collective operations.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to sync.
    #[allow(unused)]
    fn sync_collective(device: &B::Device) {
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
