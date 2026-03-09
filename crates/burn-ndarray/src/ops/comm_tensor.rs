use std::collections::HashMap;

use burn_backend::{
    ReduceOperation, ShardedParams, close_gradient_sync_server, get_gradient_sync_client,
    ops::{CommunicationTensorOps, TensorRef},
    start_gradient_sync_server,
};

use crate::{
    FloatNdArrayElement, IntNdArrayElement, NdArray, NdArrayDevice, NdArrayTensor, QuantElement,
    SharedArray,
};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> CommunicationTensorOps<Self>
    for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    // fn start_communication_server(devices: Vec<NdArrayDevice>) {
    //     start_gradient_sync_server::<Self>(devices);
    // }

    // fn close_communication_server(device: &NdArrayDevice) {
    //     close_gradient_sync_server::<Self>(device);
    // }

    // fn register_graph(
    //     device: &NdArrayDevice,
    //     n_required_map: HashMap<u64, usize>,
    //     sharded_params_map: HashMap<u64, ShardedParams>,
    // ) {
    //     if let Some(sync_client) = get_gradient_sync_client::<Self>(device) {
    //         sync_client.register_device(n_required_map, sharded_params_map);
    //     };
    // }

    // /// Wait for the queued communication operations to be finished.
    // /// TODO: ARGS and returns
    // fn communication_sync() {
    //     unimplemented!()
    // }

    // fn all_reduce_inplace(
    //     tensor: TensorRef<Self>,
    //     peer_id: burn_backend::PeerId,
    //     _all_ids: Vec<burn_backend::PeerId>,
    //     op: ReduceOperation,
    // ) {
    //     // TODO: Server
    //     if let Some(sync_client) = get_gradient_sync_client::<Self>(&Self::comm_device(&tensor)) {
    //         println!("Got client!");
    //         // sync_client.on_register(NodeId::from(0), tensor);
    //     };
    //     // println!("ndarray all_reduce");
    //     // let result = all_reduce::<Self>(peer_id, Self::float_data_from_comm(&tensor), op).unwrap();
    //     // println!("ndarray all_reduce DONE");
    //     // unsafe {
    //     //     (**tensor.0) = result;
    //     // }
    // }
}
