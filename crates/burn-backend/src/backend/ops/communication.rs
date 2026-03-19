use std::sync::Arc;

use crate::{
    Backend, DistributedParams, PeerId, ReduceOperation, close_gradient_sync_server,
    get_gradient_sync_client, start_gradient_sync_server,
    tensor::{Device, FloatTensor},
};

/// mutable reference to a float tensor.
#[derive(Clone)]
pub struct TensorRef<B: Backend>(pub Arc<*mut FloatTensor<B>>);
unsafe impl<B> Sync for TensorRef<B> where B: Backend {}
unsafe impl<B> Send for TensorRef<B> where B: Backend {}

/// Operations on communication tensors.
pub trait CommunicationTensorOps<B: Backend> {
    /// Start the communication server used to orchestrate tensor syncing between devices.
    ///
    /// # Arguments
    ///
    /// * `devices` - The devices to orchestrate.
    fn start_communication_server(devices: Vec<B::Device>) {
        start_gradient_sync_server::<B>(devices);
    }

    /// Close the communication server used to orchestrate syncing between devices.
    ///
    /// # Arguments
    ///
    /// * `device` - A device on the backend.
    fn close_communication_server(_device: &B::Device) {
        close_gradient_sync_server::<B>();
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
        if let Some(sync_client) = get_gradient_sync_client::<B>() {
            sync_client.register_device(distributed_params);
        };
    }

    /// Wait for all queued collective operations to be finished.
    ///
    /// # Arguments
    ///
    /// * `device` - The device on which to sync.
    fn sync_collective(device: &B::Device) {
        if let Some(sync_client) = get_gradient_sync_client::<B>() {
            sync_client.wait_gradients_sync(device.clone());
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
    fn submit_gradient_sync(tensor: TensorRef<B>, distributed_params: DistributedParams) {
        if let Some(sync_client) = get_gradient_sync_client::<B>() {
            sync_client.submit_gradient_sync(tensor, distributed_params);
        };
    }

    /// Whether this backend supports collective operations natively e.g. NCCL for Cuda.
    #[allow(unused)]
    fn supports_native_collective(device: &B::Device) -> bool {
        false
    }

    /// In-place version of all_reduce.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor on which to perform all_reduce.
    /// * `peer_id` - The [PeerId] of the device the tensor is on.
    /// * `all_ids` - The [PeerId] of the devices on which to all_reduce.
    /// * `op` - The [`ReduceOperation`].
    #[allow(unused)]
    fn all_reduce_in_place_native(
        tensor: TensorRef<B>,
        peer_id: PeerId,
        all_ids: Vec<PeerId>,
        op: ReduceOperation,
    ) {
        unimplemented!()
    }

    /// Natively sync the collective operations.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to sync.
    #[allow(unused)]
    fn collective_sync_native(device: &B::Device) {
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
    unsafe fn comm_device(tensor: &TensorRef<B>) -> Device<B> {
        unsafe { B::float_device(&(**tensor.0)) }
    }

    /// Creates a float tensor from the current data in the communication tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// A float tensor containing a copy of the data of the given tensor.
    unsafe fn float_data_from_comm(tensor: &TensorRef<B>) -> FloatTensor<B> {
        unsafe { (**tensor.0).clone() }
    }
}
