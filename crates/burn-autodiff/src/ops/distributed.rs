use alloc::vec::Vec;
use burn_backend::{
    DeviceId,
    distributed::{
        CollectiveTensor, DistributedBackend, DistributedConfig, DistributedParams,
        ReduceOperation, TensorRef,
    },
    tensor::FloatTensor,
};

use crate::{
    Autodiff,
    checkpoint::strategy::CheckpointStrategy,
    ops::{Backward, Ops, OpsKind, unary},
};

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
        #[derive(Debug)]
        struct AllReduce;

        impl<B: DistributedBackend> Backward<B, 1> for AllReduce {
            type State = (ReduceOperation, Vec<DeviceId>);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut crate::grads::Gradients,
                _checkpointer: &mut crate::checkpoint::base::Checkpointer,
            ) {
                // Backward uses the same reduce op: local gradients are synchronized via the backend, which handles
                // scaling (e.g., ncclAvg for mean). This works for the reduce ops that are currently supported, but we
                // might need to rework it if we add other ops such as ncclMax.
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    B::all_reduce(grad, ops.state.0, ops.state.1).resolve()
                });
            }
        }

        let collective = B::all_reduce(tensor.primitive, op, device_ids.clone());
        // Safety: we call `assume_resolved` only to wrap it in a new `CollectiveTensor`.
        let resolved = unsafe { collective.assume_resolved() };

        match AllReduce
            .prepare::<C>([tensor.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(preps) => {
                let output = preps.finish((op, device_ids), resolved);
                CollectiveTensor::new(output)
            }
            OpsKind::UnTracked(preps) => {
                let output = preps.finish(resolved);
                CollectiveTensor::new(output)
            }
        }
    }

    fn sync_collective(device: &B::Device) {
        B::sync_collective(device);
    }
}
