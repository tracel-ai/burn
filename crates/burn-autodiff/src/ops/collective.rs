use alloc::vec::Vec;
use burn_backend::{
    Backend,
    distributed::ReduceOperation,
    ops::CollectiveOps,
    tensor::{BoolTensor, Device, IntTensor},
};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy};

// Float collectives compose from tracked ops so gradients flow through them; ints and bools carry
// no gradient, so they run on the inner backend and keep its transport.
impl<B: Backend, C: CheckpointStrategy> CollectiveOps<Self> for Autodiff<B, C> {
    fn int_all_reduce(tensors: Vec<IntTensor<Self>>, op: ReduceOperation) -> Vec<IntTensor<Self>> {
        B::int_all_reduce(tensors, op)
    }

    fn int_all_gather(tensors: Vec<IntTensor<Self>>, dim: usize) -> Vec<IntTensor<Self>> {
        B::int_all_gather(tensors, dim)
    }

    fn int_reduce_scatter(
        tensors: Vec<IntTensor<Self>>,
        dim: usize,
        op: ReduceOperation,
    ) -> Vec<IntTensor<Self>> {
        B::int_reduce_scatter(tensors, dim, op)
    }

    fn int_all_to_all(
        tensors: Vec<IntTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<IntTensor<Self>> {
        B::int_all_to_all(tensors, split_dim, concat_dim)
    }

    fn int_broadcast(tensor: IntTensor<Self>, devices: &[Device<Self>]) -> Vec<IntTensor<Self>> {
        B::int_broadcast(tensor, devices)
    }

    fn bool_all_gather(tensors: Vec<BoolTensor<Self>>, dim: usize) -> Vec<BoolTensor<Self>> {
        B::bool_all_gather(tensors, dim)
    }

    fn bool_all_to_all(
        tensors: Vec<BoolTensor<Self>>,
        split_dim: usize,
        concat_dim: usize,
    ) -> Vec<BoolTensor<Self>> {
        B::bool_all_to_all(tensors, split_dim, concat_dim)
    }

    fn bool_broadcast(tensor: BoolTensor<Self>, devices: &[Device<Self>]) -> Vec<BoolTensor<Self>> {
        B::bool_broadcast(tensor, devices)
    }
}
