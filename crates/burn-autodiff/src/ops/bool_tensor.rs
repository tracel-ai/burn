use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy, tensor::AutodiffTensor};
use alloc::vec::Vec;

use burn_backend::{
    Backend, ExecutionError, TensorData,
    ops::BoolTensorOps,
    tensor::{BoolTensor, Device, IntTensor},
};
use burn_std::Shape;

impl<B: Backend, C: CheckpointStrategy> BoolTensorOps<Self> for Autodiff<B, C> {
    fn bool_from_data(data: TensorData, device: &Device<B>) -> BoolTensor<B> {
        B::bool_from_data(data, device)
    }

    async fn bool_into_data(tensor: BoolTensor<B>) -> Result<TensorData, ExecutionError> {
        B::bool_into_data(tensor).await
    }

    fn bool_into_int(tensor: BoolTensor<B>) -> IntTensor<B> {
        B::bool_into_int(tensor)
    }

    fn bool_to_device(tensor: BoolTensor<B>, device: &Device<B>) -> BoolTensor<B> {
        B::bool_to_device(tensor, device)
    }

    fn bool_device(tensor: &BoolTensor<B>) -> Device<B> {
        B::bool_device(tensor)
    }

    fn bool_reshape(tensor: BoolTensor<B>, shape: Shape) -> BoolTensor<B> {
        B::bool_reshape(tensor, shape)
    }

    fn bool_slice(tensor: BoolTensor<B>, slices: &[burn_std::Slice]) -> BoolTensor<B> {
        B::bool_slice(tensor, slices)
    }

    fn bool_empty(shape: Shape, device: &Device<B>) -> BoolTensor<B> {
        B::bool_empty(shape, device)
    }

    fn bool_zeros(shape: Shape, device: &Device<B>) -> BoolTensor<B> {
        B::bool_zeros(shape, device)
    }

    fn bool_ones(shape: Shape, device: &Device<B>) -> BoolTensor<B> {
        B::bool_ones(shape, device)
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        slices: &[burn_std::Slice],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        B::bool_slice_assign(tensor, slices, value)
    }

    fn bool_cat(tensors: Vec<BoolTensor<B>>, dim: usize) -> BoolTensor<B> {
        B::bool_cat(tensors, dim)
    }

    fn bool_equal(lhs: BoolTensor<B>, rhs: BoolTensor<B>) -> BoolTensor<B> {
        B::bool_equal(lhs, rhs)
    }

    fn bool_not(tensor: BoolTensor<B>) -> BoolTensor<B> {
        B::bool_not(tensor)
    }

    fn bool_and(lhs: BoolTensor<B>, rhs: BoolTensor<B>) -> BoolTensor<B> {
        B::bool_and(lhs, rhs)
    }

    fn bool_or(lhs: BoolTensor<B>, rhs: BoolTensor<B>) -> BoolTensor<B> {
        B::bool_or(lhs, rhs)
    }

    fn bool_xor(lhs: BoolTensor<B>, rhs: BoolTensor<B>) -> BoolTensor<B> {
        B::bool_xor(lhs, rhs)
    }

    fn bool_into_float(tensor: BoolTensor<B>) -> <Autodiff<B> as Backend>::FloatTensorPrimitive {
        AutodiffTensor::new(B::bool_into_float(tensor))
    }

    fn bool_swap_dims(
        tensor: <Autodiff<B> as Backend>::BoolTensorPrimitive,
        dim1: usize,
        dim2: usize,
    ) -> <Autodiff<B> as Backend>::BoolTensorPrimitive {
        B::bool_swap_dims(tensor, dim1, dim2)
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        B::bool_permute(tensor, axes)
    }

    fn bool_flip(tensor: BoolTensor<B>, axes: &[usize]) -> BoolTensor<B> {
        B::bool_flip(tensor, axes)
    }

    async fn bool_argwhere(tensor: BoolTensor<B>) -> IntTensor<B> {
        B::bool_argwhere(tensor).await
    }

    fn bool_expand(tensor: BoolTensor<B>, shape: Shape) -> BoolTensor<B> {
        B::bool_expand(tensor, shape)
    }

    fn bool_repeat_dim(tensor: BoolTensor<B>, dim: usize, times: usize) -> BoolTensor<B> {
        B::bool_repeat_dim(tensor, dim, times)
    }

    fn bool_unfold(
        tensor: BoolTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<Self> {
        B::bool_unfold(tensor, dim, size, step)
    }
}
