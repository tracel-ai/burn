use alloc::vec::Vec;
use burn_backend::distributed::{
    DistributedBackend, DistributedConfig, DistributedParams, ReduceOperation, TensorRef,
};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy};

impl<B: DistributedBackend, C: CheckpointStrategy> DistributedBackend for Autodiff<B, C> {
    fn start_communication_server(devices: &[B::Device], config: DistributedConfig) {
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

    fn submit_gradient_sync(tensor: TensorRef<Self>, distributed_params: DistributedParams) {
        let mut tensor = unsafe { (*tensor.0).clone() };
        B::submit_gradient_sync(TensorRef(&mut tensor.primitive), distributed_params);
    }

    unsafe fn all_reduce_in_place(tensors: Vec<TensorRef<Self>>, op: ReduceOperation) {
        let tensors = tensors
            .iter()
            .map(|t| {
                let mut t = unsafe { (*t.0).clone() };
                TensorRef(&mut t.primitive)
            })
            .collect();
        unsafe { B::all_reduce_in_place(tensors, op) };
    }

    fn sync_collective(device: &B::Device) {
        B::sync_collective(device);
    }
}
