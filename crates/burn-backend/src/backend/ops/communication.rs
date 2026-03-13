use std::sync::Arc;

use crate::{
    Backend, PeerId, ReduceOperation, ShardedParams, close_gradient_sync_server,
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
    /// Start the communication server used to orchestrate operations between devices.
    ///
    /// # Arguments
    ///
    /// * `devices` - The devices to orchestrate.
    fn start_communication_server(devices: Vec<B::Device>) {
        start_gradient_sync_server::<B>(devices);
    }

    /// Close the communication server used to orchestrate operations between devices.
    ///
    /// # Arguments
    ///
    /// * `device` - A device on the backend.
    fn close_communication_server(_device: &B::Device) {
        close_gradient_sync_server::<B>();
    }

    /// Announce the parameters to sync during a single backward pass on this device to the gradient sync server of the backend.
    ///
    /// # Arguments
    ///
    /// * `device` - The device calling the initialization.
    /// * `sharded_param_ids` - A list of [`ShardedParams`] of the tensors to sync.
    fn init_collective_queue(_device: &B::Device, sharded_param_ids: Vec<ShardedParams>) {
        if let Some(sync_client) = get_gradient_sync_client::<B>() {
            sync_client.register_device(sharded_param_ids);
        };
    }

    /// Wait for the all queued collective operations to be finished.
    ///
    /// # Arguments
    ///
    /// * `device` - The device on which to sync.
    fn collective_sync(device: &B::Device) {
        if let Some(sync_client) = get_gradient_sync_client::<B>() {
            sync_client.wait_gradients_sync(device.clone());
        };
    }

    /// Performs an in place all_reduce operation on the given sharded tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor on which to perform all_reduce.
    /// * `sharded_params` - The [`ShardedParams`].
    fn all_reduce_in_place(tensor: TensorRef<B>, sharded_params: ShardedParams) {
        if let Some(sync_client) = get_gradient_sync_client::<B>() {
            sync_client.on_register(tensor, sharded_params);
        };
    }

    /// Whether this backend supports collective operations natively e.g. NCCL for Cuda.
    #[allow(unused)]
    fn supports_native_collective(device: &B::Device) -> bool {
        false
    }

    /// The native version of the all_reduce.
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
        unimplemented!()
    }

    /////////////////////////////////////////////////////////////TODO: useful?//////////////////////////////////////////////////////////////

    /// Performs a broadcast of the given source tensor to the destinations, in-place.
    ///
    /// # Arguments
    ///
    /// * `src_tensors` - A float tensor of the data to broadcast.
    /// * `dest_tensors` - The tensors on which to perform the broadcast in-place.
    fn all_broadcast_inplace(src_tensor: FloatTensor<B>, dest_tensors: Vec<TensorRef<B>>) {
        unsafe {
            for dest in dest_tensors {
                let device = B::comm_device(&dest);
                let tensor_float = B::float_to_device(src_tensor.clone(), &device);
                (**dest.0) = tensor_float;
            }
        }
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
    fn comm_device(tensor: &TensorRef<B>) -> Device<B> {
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
    fn float_data_from_comm(tensor: &TensorRef<B>) -> FloatTensor<B> {
        unsafe { (**tensor.0).clone() }
    }
}
