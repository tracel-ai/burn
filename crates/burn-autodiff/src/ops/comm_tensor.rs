use burn_backend::{
    Backend, DistributedParams, PeerId, ReduceOperation,
    ops::{CommunicationTensorOps, TensorRef},
};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy};

impl<B: Backend, C: CheckpointStrategy> CommunicationTensorOps<Self> for Autodiff<B, C> {
    fn start_communication_server(devices: Vec<B::Device>) {
        B::start_communication_server(devices);
    }

    fn close_communication_server(device: &B::Device) {
        B::close_communication_server(device);
    }

    fn register_sync_parameters(device: &B::Device, distributed_params: Vec<DistributedParams>) {
        B::register_sync_parameters(device, distributed_params);
    }

    fn sync_collective(device: &B::Device) {
        B::sync_collective(device);
    }

    fn submit_gradient_sync(_tensor: TensorRef<Self>, _distributed_params: DistributedParams) {
        unimplemented!()
    }

    fn supports_native_collective(device: &B::Device) -> bool {
        B::supports_native_collective(device)
    }

    fn all_reduce_in_place_native(
        _tensor: TensorRef<Self>,
        _peer_id: PeerId,
        _all_ids: Vec<PeerId>,
        _op: ReduceOperation,
    ) {
        unimplemented!()
    }

    #[allow(unused)]
    fn collective_sync_native(device: &B::Device) {
        unimplemented!()
    }
}
