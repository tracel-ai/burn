use alloc::vec::Vec;
use burn_backend::{
    Backend,
    distributed::{DistributedConfig, DistributedOps, DistributedParams, TensorRef},
};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy};

impl<B: Backend, C: CheckpointStrategy> DistributedOps<Self> for Autodiff<B, C> {
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
}
