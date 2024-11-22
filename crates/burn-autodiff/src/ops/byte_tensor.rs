use crate::{checkpoint::strategy::CheckpointStrategy, tensor::AutodiffTensor, Autodiff};

use burn_tensor::{
    backend::Backend,
    ops::{BoolTensor, ByteTensor, ByteTensorOps, IntTensor},
    Device, Distribution, Shape, TensorData,
};

impl<B: Backend, C: CheckpointStrategy> ByteTensorOps<Self> for Autodiff<B, C> {
    fn byte_from_data(data: TensorData, device: &Device<Self>) -> ByteTensor<B> {
        B::byte_from_data(data, device)
    }

    async fn byte_into_data(tensor: ByteTensor<B>) -> TensorData {
        B::byte_into_data(tensor).await
    }

    fn byte_to_device(tensor: ByteTensor<B>, device: &Device<Self>) -> ByteTensor<B> {
        B::byte_to_device(tensor, device)
    }

    fn byte_device(tensor: &ByteTensor<B>) -> Device<Self> {
        B::byte_device(tensor)
    }

    fn byte_reshape(tensor: ByteTensor<B>, shape: Shape) -> ByteTensor<B> {
        B::byte_reshape(tensor, shape)
    }

    fn byte_slice(tensor: ByteTensor<B>, ranges: &[std::ops::Range<usize>]) -> ByteTensor<B> {
        B::byte_slice(tensor, ranges)
    }

    fn byte_empty(shape: Shape, device: &<Autodiff<B> as Backend>::Device) -> ByteTensor<B> {
        B::byte_empty(shape, device)
    }

    fn byte_slice_assign(
        tensor: ByteTensor<B>,
        ranges: &[std::ops::Range<usize>],
        value: ByteTensor<B>,
    ) -> ByteTensor<B> {
        B::byte_slice_assign(tensor, ranges, value)
    }

    fn byte_cat(tensors: Vec<ByteTensor<B>>, dim: usize) -> ByteTensor<B> {
        B::byte_cat(tensors, dim)
    }

    fn byte_equal(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> BoolTensor<B> {
        B::byte_equal(lhs, rhs)
    }

    fn byte_equal_elem(lhs: ByteTensor<B>, rhs: B::ByteElem) -> BoolTensor<B> {
        B::byte_equal_elem(lhs, rhs)
    }

    fn byte_add(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> ByteTensor<B> {
        B::byte_add(lhs, rhs)
    }

    fn byte_add_scalar(lhs: ByteTensor<B>, rhs: B::ByteElem) -> ByteTensor<B> {
        B::byte_add_scalar(lhs, rhs)
    }

    fn byte_clamp_min(tensor: ByteTensor<B>, min: B::ByteElem) -> ByteTensor<B> {
        B::byte_clamp_min(tensor, min)
    }

    fn byte_clamp_max(tensor: ByteTensor<B>, max: B::ByteElem) -> ByteTensor<B> {
        B::byte_clamp_max(tensor, max)
    }

    fn byte_clamp(tensor: ByteTensor<B>, min: B::ByteElem, max: B::ByteElem) -> ByteTensor<B> {
        B::byte_clamp(tensor, min, max)
    }

    fn byte_sub(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> ByteTensor<B> {
        B::byte_sub(lhs, rhs)
    }

    fn byte_sub_scalar(lhs: ByteTensor<B>, rhs: B::ByteElem) -> ByteTensor<B> {
        B::byte_sub_scalar(lhs, rhs)
    }

    fn byte_mul(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> ByteTensor<B> {
        B::byte_mul(lhs, rhs)
    }

    fn byte_mul_scalar(lhs: ByteTensor<B>, rhs: B::ByteElem) -> ByteTensor<B> {
        B::byte_mul_scalar(lhs, rhs)
    }

    fn byte_div(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> ByteTensor<B> {
        B::byte_div(lhs, rhs)
    }

    fn byte_div_scalar(lhs: ByteTensor<B>, rhs: B::ByteElem) -> ByteTensor<B> {
        B::byte_div_scalar(lhs, rhs)
    }

    fn byte_remainder(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> ByteTensor<B> {
        B::byte_remainder(lhs, rhs)
    }

    fn byte_remainder_scalar(lhs: ByteTensor<B>, rhs: B::ByteElem) -> ByteTensor<B> {
        B::byte_remainder_scalar(lhs, rhs)
    }

    fn byte_neg(tensor: ByteTensor<B>) -> ByteTensor<B> {
        B::byte_neg(tensor)
    }

    fn byte_zeros(shape: Shape, device: &Device<Self>) -> ByteTensor<B> {
        B::byte_zeros(shape, device)
    }

    fn byte_ones(shape: Shape, device: &Device<Self>) -> ByteTensor<B> {
        B::byte_ones(shape, device)
    }

    fn byte_full(shape: Shape, fill_value: B::ByteElem, device: &Device<Self>) -> ByteTensor<B> {
        B::byte_full(shape, fill_value, device)
    }

    fn byte_sum(tensor: ByteTensor<B>) -> ByteTensor<B> {
        B::byte_sum(tensor)
    }

    fn byte_sum_dim(tensor: ByteTensor<B>, dim: usize) -> ByteTensor<B> {
        B::byte_sum_dim(tensor, dim)
    }

    fn byte_mean(tensor: ByteTensor<B>) -> ByteTensor<B> {
        B::byte_mean(tensor)
    }

    fn byte_mean_dim(tensor: ByteTensor<B>, dim: usize) -> ByteTensor<B> {
        B::byte_mean_dim(tensor, dim)
    }

    fn byte_repeat_dim(tensor: ByteTensor<B>, dim: usize, times: usize) -> ByteTensor<B> {
        B::byte_repeat_dim(tensor, dim, times)
    }

    fn byte_greater(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> BoolTensor<B> {
        B::byte_greater(lhs, rhs)
    }

    fn byte_greater_elem(lhs: ByteTensor<B>, rhs: B::ByteElem) -> BoolTensor<B> {
        B::byte_greater_elem(lhs, rhs)
    }

    fn byte_greater_equal(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> BoolTensor<B> {
        B::byte_greater_equal(lhs, rhs)
    }

    fn byte_greater_equal_elem(lhs: ByteTensor<B>, rhs: B::ByteElem) -> BoolTensor<B> {
        B::byte_greater_equal_elem(lhs, rhs)
    }

    fn byte_lower(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> BoolTensor<B> {
        B::byte_lower(lhs, rhs)
    }

    fn byte_lower_elem(lhs: ByteTensor<B>, rhs: B::ByteElem) -> BoolTensor<B> {
        B::byte_lower_elem(lhs, rhs)
    }

    fn byte_lower_equal(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> BoolTensor<B> {
        B::byte_lower_equal(lhs, rhs)
    }

    fn byte_lower_equal_elem(lhs: ByteTensor<B>, rhs: B::ByteElem) -> BoolTensor<B> {
        B::byte_lower_equal_elem(lhs, rhs)
    }

    fn byte_gather(dim: usize, tensor: ByteTensor<B>, indices: IntTensor<B>) -> ByteTensor<B> {
        B::byte_gather(dim, tensor, indices)
    }

    fn byte_scatter(
        dim: usize,
        tensor: ByteTensor<B>,
        indices: IntTensor<B>,
        value: ByteTensor<B>,
    ) -> ByteTensor<B> {
        B::byte_scatter(dim, tensor, indices, value)
    }

    fn byte_select(tensor: ByteTensor<B>, dim: usize, indices: IntTensor<B>) -> ByteTensor<B> {
        B::byte_select(tensor, dim, indices)
    }

    fn byte_select_assign(
        tensor: ByteTensor<B>,
        dim: usize,
        indices: IntTensor<B>,
        value: ByteTensor<B>,
    ) -> ByteTensor<B> {
        B::byte_select_assign(tensor, dim, indices, value)
    }

    fn byte_mask_where(
        tensor: ByteTensor<B>,
        mask: BoolTensor<B>,
        value: ByteTensor<B>,
    ) -> <Autodiff<B> as Backend>::ByteTensorPrimitive {
        B::byte_mask_where(tensor, mask, value)
    }

    fn byte_mask_fill(
        tensor: ByteTensor<B>,
        mask: BoolTensor<B>,
        value: B::ByteElem,
    ) -> <Autodiff<B> as Backend>::ByteTensorPrimitive {
        B::byte_mask_fill(tensor, mask, value)
    }

    fn byte_argmax(tensor: ByteTensor<B>, dim: usize) -> IntTensor<B> {
        B::byte_argmax(tensor, dim)
    }
    fn byte_argmin(tensor: ByteTensor<B>, dim: usize) -> IntTensor<B> {
        B::byte_argmin(tensor, dim)
    }
    fn byte_max(tensor: B::ByteTensorPrimitive) -> B::ByteTensorPrimitive {
        B::byte_max(tensor)
    }
    fn byte_max_dim(tensor: B::ByteTensorPrimitive, dim: usize) -> B::ByteTensorPrimitive {
        B::byte_max_dim(tensor, dim)
    }
    fn byte_max_dim_with_indices(
        tensor: B::ByteTensorPrimitive,
        dim: usize,
    ) -> (B::ByteTensorPrimitive, B::IntTensorPrimitive) {
        B::byte_max_dim_with_indices(tensor, dim)
    }
    fn byte_min(tensor: B::ByteTensorPrimitive) -> B::ByteTensorPrimitive {
        B::byte_min(tensor)
    }
    fn byte_min_dim(tensor: B::ByteTensorPrimitive, dim: usize) -> B::ByteTensorPrimitive {
        B::byte_min_dim(tensor, dim)
    }
    fn byte_min_dim_with_indices(
        tensor: B::ByteTensorPrimitive,
        dim: usize,
    ) -> (B::ByteTensorPrimitive, B::IntTensorPrimitive) {
        B::byte_min_dim_with_indices(tensor, dim)
    }
    fn byte_abs(tensor: B::ByteTensorPrimitive) -> B::ByteTensorPrimitive {
        B::byte_abs(tensor)
    }
    fn byte_into_float(
        tensor: <Autodiff<B> as Backend>::ByteTensorPrimitive,
    ) -> <Autodiff<B> as Backend>::FloatTensorPrimitive {
        AutodiffTensor::new(B::byte_into_float(tensor))
    }
    fn byte_into_int(
        tensor: <Autodiff<B> as Backend>::ByteTensorPrimitive,
    ) -> <Autodiff<B> as Backend>::IntTensorPrimitive {
        B::byte_into_int(tensor)
    }

    fn byte_swap_dims(
        tensor: <Autodiff<B> as Backend>::ByteTensorPrimitive,
        dim1: usize,
        dim2: usize,
    ) -> <Autodiff<B> as Backend>::ByteTensorPrimitive {
        B::byte_swap_dims(tensor, dim1, dim2)
    }

    fn byte_narrow(
        tensor: <Autodiff<B> as Backend>::ByteTensorPrimitive,
        dim: usize,
        start: usize,
        length: usize,
    ) -> <Autodiff<B> as Backend>::ByteTensorPrimitive {
        B::byte_narrow(tensor, dim, start, length)
    }

    fn byte_chunk(
        tensor: <Autodiff<B> as Backend>::ByteTensorPrimitive,
        chunks: usize,
        dim: usize,
    ) -> Vec<<Autodiff<B> as Backend>::ByteTensorPrimitive> {
        B::byte_chunk(tensor, chunks, dim)
    }

    fn byte_split(
        tensor: <Autodiff<B> as Backend>::ByteTensorPrimitive,
        split_size: usize,
        dim: usize,
    ) -> Vec<<Autodiff<B> as Backend>::ByteTensorPrimitive> {
        B::byte_split(tensor, split_size, dim)
    }

    fn byte_split_with_sizes(
        tensor: <Autodiff<B> as Backend>::ByteTensorPrimitive,
        split_sizes: Vec<usize>,
        dim: usize,
    ) -> Vec<<Autodiff<B> as Backend>::ByteTensorPrimitive> {
        B::byte_split_with_sizes(tensor, split_sizes, dim)
    }

    fn byte_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> ByteTensor<Self> {
        B::byte_random(shape, distribution, device)
    }

    fn byte_arange(range: std::ops::Range<i64>, device: &Device<Self>) -> ByteTensor<Self> {
        B::byte_arange(range, device)
    }

    fn byte_permute(tensor: ByteTensor<Self>, axes: &[usize]) -> ByteTensor<Self> {
        B::byte_permute(tensor, axes)
    }

    fn byte_flip(tensor: ByteTensor<Self>, axes: &[usize]) -> ByteTensor<Self> {
        B::byte_flip(tensor, axes)
    }

    fn byte_sign(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        B::byte_sign(tensor)
    }

    fn byte_prod(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        B::byte_prod(tensor)
    }

    fn byte_prod_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        B::byte_prod_dim(tensor, dim)
    }

    fn byte_expand(tensor: ByteTensor<B>, shape: Shape) -> ByteTensor<B> {
        B::byte_expand(tensor, shape)
    }

    fn byte_sort(tensor: ByteTensor<Self>, dim: usize, descending: bool) -> ByteTensor<Self> {
        B::byte_sort(tensor, dim, descending)
    }

    fn byte_sort_with_indices(
        tensor: ByteTensor<Self>,
        dim: usize,
        descending: bool,
    ) -> (ByteTensor<Self>, IntTensor<Self>) {
        B::byte_sort_with_indices(tensor, dim, descending)
    }

    fn byte_argsort(tensor: ByteTensor<Self>, dim: usize, descending: bool) -> IntTensor<Self> {
        B::byte_argsort(tensor, dim, descending)
    }
}
