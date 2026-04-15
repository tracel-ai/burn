use alloc::vec::Vec;
use burn_backend::{
    DeviceId,
    distributed::{
        CollectiveTensor, DistributedBackend, DistributedConfig, DistributedParams,
        ReduceOperation, TensorRef,
    },
    tensor::FloatTensor,
};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy, tensor::AutodiffTensor};

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

    fn all_reduce(
        tensor: FloatTensor<Self>,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> CollectiveTensor<Self> {
        // TODO: backward()
        let tensor = B::all_reduce(tensor.primitive, op, device_ids);
        // Safety: we call `assume_resolved` only to wrap it in a new `CollectiveTensor`.
        CollectiveTensor::new(AutodiffTensor::new(unsafe { tensor.assume_resolved() }))
    }

    fn sync_collective(device: &B::Device) {
        B::sync_collective(device);
    }
}
