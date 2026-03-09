use std::{collections::HashMap, sync::Arc};

use crate::{
    Backend, ModuleParamId, PeerId, ReduceOperation, ShardedParams, close_gradient_sync_server,
    get_gradient_sync_client, start_gradient_sync_server,
    tensor::{Device, FloatTensor},
};

pub(crate) unsafe fn reduce_sum_centralized<B: Backend>(
    tensors: &Vec<TensorRef<B>>,
    central_device: &B::Device,
) -> B::FloatTensorPrimitive {
    let mut central_tensor = (**tensors[0].0).clone();
    for tensor in tensors {
        let rhs = B::float_to_device((**tensor.0).clone(), &central_device);
        central_tensor = B::float_add(central_tensor, rhs);
    }

    central_tensor
}

pub(crate) unsafe fn all_reduce_inplace_sum_centralized<B: Backend>(
    tensors: Vec<TensorRef<B>>,
    op: ReduceOperation,
) {
    let devices: Vec<B::Device> = tensors
        .iter()
        .map(|tensor| B::comm_device(tensor))
        .collect();
    let central_device = devices.get(0).unwrap();

    // TODO: inplace?
    let mut central_tensor = reduce_sum_centralized::<B>(&tensors, &central_device);

    if op == ReduceOperation::Mean {
        // Apply mean division
        let div = (tensors.len() as f32).into();
        central_tensor = B::float_div_scalar(central_tensor, div);
    }

    // Broadcast result to all
    B::all_broadcast_inplace(central_tensor, tensors);
}

#[derive(Clone)]
pub struct TensorRef<B: Backend>(pub Arc<*mut FloatTensor<B>>);
unsafe impl<B> Sync for TensorRef<B> where B: Backend {}
unsafe impl<B> Send for TensorRef<B> where B: Backend {}

/// Operations on communication tensors.
pub trait CommunicationTensorOps<B: Backend> {
    /// Start the communication server used to orchestrate operations across devices.
    ///
    /// # Arguments
    ///
    /// * `devices` - The devices to orchestrate.
    fn start_communication_server(devices: Vec<B::Device>) {
        start_gradient_sync_server::<B>(devices);
    }

    /// Close the communication server used to orchestrate operations across devices.
    ///
    /// # Arguments
    ///
    /// * `devices` - The devices to orchestrate.
    fn close_communication_server(device: &B::Device) {
        close_gradient_sync_server::<B>(device);
    }

    /// Register the maps for an autodiff graph of a backward pass.
    /// TODO: ARGS and returns
    fn register_graph(
        device: &B::Device,
        // n_required_map: HashMap<u64, usize>,
        // sharded_params_map: HashMap<u64, ShardedParams>,
        sharded_param_ids: Vec<ShardedParams>,
    ) {
        if let Some(sync_client) = get_gradient_sync_client::<B>(device) {
            sync_client.register_device(sharded_param_ids);
            // sync_client.register_device(n_required_map, sharded_params_map);
        };
    }

    /// Wait for the queued communication operations to be finished.
    /// TODO: ARGS and returns
    fn communication_sync(device: &B::Device) {
        if let Some(sync_client) = get_gradient_sync_client::<B>(device) {
            println!("comm sync");
            // sync_client.register_device(n_required_map, sharded_params_map);
        };
    }

    /// Performs an all_reduce operation on the given tensors and replaces the values in-place.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor on which to perform all_reduce.
    /// * `peer_id` - The device's [PeerId].
    /// * `all_ids` - All of the devices [PeerId]'s from which to all_reduce.
    /// * `op` - The [`ReduceOperation`].
    fn all_reduce_inplace(tensor: TensorRef<B>, sharded_params: ShardedParams) {
        if let Some(sync_client) = get_gradient_sync_client::<B>(&B::comm_device(&tensor)) {
            sync_client.on_register(tensor, sharded_params);
        };
    }

    /// If this backend supports native communication operations e.g. NCCL for Cuda.
    /// TODO: ARGS and returns
    fn supports_native_communication(_device: &B::Device) -> bool {
        false
    }

    fn all_reduce_inplace_native(
        _tensor: TensorRef<B>,
        _peer_id: PeerId,
        _all_ids: Vec<PeerId>,
        _op: ReduceOperation,
    ) {
        unimplemented!()
    }

    // unsafe fn all_reduce_inplace(
    //     tensors: Vec<TensorRef<B>>,
    //     strategy: AllReduceStrategy,
    //     op: ReduceOperation,
    // ) {
    //     match strategy {
    //         AllReduceStrategy::Centralized => all_reduce_inplace_sum_centralized::<B>(tensors, op),
    //         // AllReduceStrategy::Tree(arity) => all_reduce_sum_tree::<B>(tensors, *arity),
    //         // AllReduceStrategy::Ring => all_reduce_sum_ring::<B>(tensors),
    //         AllReduceStrategy::Tree(arity) => todo!(),
    //         AllReduceStrategy::Ring => todo!(),
    //     };
    // }

    /// Performs a broadcast of the given source tensor to the destinations, in-place.
    ///
    /// # Arguments
    ///
    /// * `src_tensors` - A float tensor of the data to broadcast.
    /// * `dest_tensors` - The tensors on which to perform the broadcast in-place.
    unsafe fn all_broadcast_inplace(src_tensor: FloatTensor<B>, dest_tensors: Vec<TensorRef<B>>) {
        for dest in dest_tensors {
            let device = B::comm_device(&dest);
            let tensor_float = B::float_to_device(src_tensor.clone(), &device);
            (**dest.0) = tensor_float;
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
