use alloc::vec::Vec;
use burn_backend::distributed::{
    DistributedBackend, DistributedConfig, DistributedParams, ReduceOperation, TensorRef,
};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy};

impl<B: DistributedBackend, C: CheckpointStrategy> DistributedBackend for Autodiff<B, C> {
    fn start_communication_server(devices: Vec<B::Device>, config: DistributedConfig) {
        B::start_communication_server(devices, config);
    }

    fn close_communication_server(device: &B::Device) {
        B::close_communication_server(device);
    }

    fn register_sync_parameters(device: &B::Device, distributed_params: Vec<DistributedParams>) {
        B::register_sync_parameters(device, distributed_params);
    }

    fn submit_sync_collective(device: &B::Device) {
        B::submit_sync_collective(device);
    }

    fn submit_gradient_sync(_tensor: TensorRef<Self>, _distributed_params: DistributedParams) {
        // Shouldn't be called on an autodiff tensor.
        unimplemented!()
    }

    unsafe fn all_reduce_in_place(_tensors: Vec<TensorRef<Self>>, _op: ReduceOperation) {
        // Shouldn't be called on an autodiff tensor.
        unimplemented!()
    }

    fn sync_collective(_device: &B::Device) {
        // Shouldn't be called for an autodiff backend.
        unimplemented!()
    }
}
