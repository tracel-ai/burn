use crate::{checkpoint::strategy::CheckpointStrategy, tensor::AutodiffTensor, Autodiff};

use burn_tensor::{
    backend::Backend,
    ops::{BoolTensor, IntTensor, IntTensorOps},
    Device, Distribution, Shape, TensorData,
};

impl<B: Backend, C: CheckpointStrategy> IntTensorOps<Self> for Autodiff<B, C> {
    fn int_from_data(data: TensorData, device: &Device<Self>) -> IntTensor<B> {
        B::int_from_data(data, device)
    }

    async fn int_into_data(tensor: IntTensor<B>) -> TensorData {
        B::int_into_data(tensor).await
    }

    fn int_to_device(tensor: IntTensor<B>, device: &Device<Self>) -> IntTensor<B> {
        B::int_to_device(tensor, device)
    }

    fn int_device(tensor: &IntTensor<B>) -> Device<Self> {
        B::int_device(tensor)
    }

    fn int_reshape(tensor: IntTensor<B>, shape: Shape) -> IntTensor<B> {
        B::int_reshape(tensor, shape)
    }

    fn int_slice(tensor: IntTensor<B>, ranges: &[std::ops::Range<usize>]) -> IntTensor<B> {
        B::int_slice(tensor, ranges)
    }

    fn int_empty(shape: Shape, device: &<Autodiff<B> as Backend>::Device) -> IntTensor<B> {
        B::int_empty(shape, device)
    }

    fn int_slice_assign(
        tensor: IntTensor<B>,
        ranges: &[std::ops::Range<usize>],
        value: IntTensor<B>,
    ) -> IntTensor<B> {
        B::int_slice_assign(tensor, ranges, value)
    }

    fn int_cat(tensors: Vec<IntTensor<B>>, dim: usize) -> IntTensor<B> {
        B::int_cat(tensors, dim)
    }

    fn int_equal(lhs: IntTensor<B>, rhs: IntTensor<B>) -> BoolTensor<B> {
        B::int_equal(lhs, rhs)
    }

    fn int_equal_elem(lhs: IntTensor<B>, rhs: B::IntElem) -> BoolTensor<B> {
        B::int_equal_elem(lhs, rhs)
    }

    fn int_add(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::int_add(lhs, rhs)
    }

    fn int_add_scalar(lhs: IntTensor<B>, rhs: B::IntElem) -> IntTensor<B> {
        B::int_add_scalar(lhs, rhs)
    }

    fn int_clamp_min(tensor: IntTensor<B>, min: B::IntElem) -> IntTensor<B> {
        B::int_clamp_min(tensor, min)
    }

    fn int_clamp_max(tensor: IntTensor<B>, max: B::IntElem) -> IntTensor<B> {
        B::int_clamp_max(tensor, max)
    }

    fn int_clamp(tensor: IntTensor<B>, min: B::IntElem, max: B::IntElem) -> IntTensor<B> {
        B::int_clamp(tensor, min, max)
    }

    fn int_sub(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::int_sub(lhs, rhs)
    }

    fn int_sub_scalar(lhs: IntTensor<B>, rhs: B::IntElem) -> IntTensor<B> {
        B::int_sub_scalar(lhs, rhs)
    }

    fn int_mul(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::int_mul(lhs, rhs)
    }

    fn int_mul_scalar(lhs: IntTensor<B>, rhs: B::IntElem) -> IntTensor<B> {
        B::int_mul_scalar(lhs, rhs)
    }

    fn int_div(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::int_div(lhs, rhs)
    }

    fn int_div_scalar(lhs: IntTensor<B>, rhs: B::IntElem) -> IntTensor<B> {
        B::int_div_scalar(lhs, rhs)
    }

    fn int_remainder(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::int_remainder(lhs, rhs)
    }

    fn int_remainder_scalar(lhs: IntTensor<B>, rhs: B::IntElem) -> IntTensor<B> {
        B::int_remainder_scalar(lhs, rhs)
    }

    fn int_neg(tensor: IntTensor<B>) -> IntTensor<B> {
        B::int_neg(tensor)
    }

    fn int_zeros(shape: Shape, device: &Device<Self>) -> IntTensor<B> {
        B::int_zeros(shape, device)
    }

    fn int_ones(shape: Shape, device: &Device<Self>) -> IntTensor<B> {
        B::int_ones(shape, device)
    }

    fn int_full(shape: Shape, fill_value: B::IntElem, device: &Device<Self>) -> IntTensor<B> {
        B::int_full(shape, fill_value, device)
    }

    fn int_sum(tensor: IntTensor<B>) -> IntTensor<B> {
        B::int_sum(tensor)
    }

    fn int_sum_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        B::int_sum_dim(tensor, dim)
    }

    fn int_mean(tensor: IntTensor<B>) -> IntTensor<B> {
        B::int_mean(tensor)
    }

    fn int_mean_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        B::int_mean_dim(tensor, dim)
    }

    fn int_repeat_dim(tensor: IntTensor<B>, dim: usize, times: usize) -> IntTensor<B> {
        B::int_repeat_dim(tensor, dim, times)
    }

    fn int_greater(lhs: IntTensor<B>, rhs: IntTensor<B>) -> BoolTensor<B> {
        B::int_greater(lhs, rhs)
    }

    fn int_greater_elem(lhs: IntTensor<B>, rhs: B::IntElem) -> BoolTensor<B> {
        B::int_greater_elem(lhs, rhs)
    }

    fn int_greater_equal(lhs: IntTensor<B>, rhs: IntTensor<B>) -> BoolTensor<B> {
        B::int_greater_equal(lhs, rhs)
    }

    fn int_greater_equal_elem(lhs: IntTensor<B>, rhs: B::IntElem) -> BoolTensor<B> {
        B::int_greater_equal_elem(lhs, rhs)
    }

    fn int_lower(lhs: IntTensor<B>, rhs: IntTensor<B>) -> BoolTensor<B> {
        B::int_lower(lhs, rhs)
    }

    fn int_lower_elem(lhs: IntTensor<B>, rhs: B::IntElem) -> BoolTensor<B> {
        B::int_lower_elem(lhs, rhs)
    }

    fn int_lower_equal(lhs: IntTensor<B>, rhs: IntTensor<B>) -> BoolTensor<B> {
        B::int_lower_equal(lhs, rhs)
    }

    fn int_lower_equal_elem(lhs: IntTensor<B>, rhs: B::IntElem) -> BoolTensor<B> {
        B::int_lower_equal_elem(lhs, rhs)
    }

    fn int_gather(dim: usize, tensor: IntTensor<B>, indices: IntTensor<B>) -> IntTensor<B> {
        B::int_gather(dim, tensor, indices)
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<B>,
        indices: IntTensor<B>,
        value: IntTensor<B>,
    ) -> IntTensor<B> {
        B::int_scatter(dim, tensor, indices, value)
    }

    fn int_select(tensor: IntTensor<B>, dim: usize, indices: IntTensor<B>) -> IntTensor<B> {
        B::int_select(tensor, dim, indices)
    }

    fn int_select_assign(
        tensor: IntTensor<B>,
        dim: usize,
        indices: IntTensor<B>,
        value: IntTensor<B>,
    ) -> IntTensor<B> {
        B::int_select_assign(tensor, dim, indices, value)
    }

    fn int_mask_where(
        tensor: IntTensor<B>,
        mask: BoolTensor<B>,
        value: IntTensor<B>,
    ) -> <Autodiff<B> as Backend>::IntTensorPrimitive {
        B::int_mask_where(tensor, mask, value)
    }

    fn int_mask_fill(
        tensor: IntTensor<B>,
        mask: BoolTensor<B>,
        value: B::IntElem,
    ) -> <Autodiff<B> as Backend>::IntTensorPrimitive {
        B::int_mask_fill(tensor, mask, value)
    }

    fn int_argmax(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        B::int_argmax(tensor, dim)
    }
    fn int_argmin(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        B::int_argmin(tensor, dim)
    }
    fn int_max(tensor: B::IntTensorPrimitive) -> B::IntTensorPrimitive {
        B::int_max(tensor)
    }
    fn int_max_dim(tensor: B::IntTensorPrimitive, dim: usize) -> B::IntTensorPrimitive {
        B::int_max_dim(tensor, dim)
    }
    fn int_max_dim_with_indices(
        tensor: B::IntTensorPrimitive,
        dim: usize,
    ) -> (B::IntTensorPrimitive, B::IntTensorPrimitive) {
        B::int_max_dim_with_indices(tensor, dim)
    }
    fn int_min(tensor: B::IntTensorPrimitive) -> B::IntTensorPrimitive {
        B::int_min(tensor)
    }
    fn int_min_dim(tensor: B::IntTensorPrimitive, dim: usize) -> B::IntTensorPrimitive {
        B::int_min_dim(tensor, dim)
    }
    fn int_min_dim_with_indices(
        tensor: B::IntTensorPrimitive,
        dim: usize,
    ) -> (B::IntTensorPrimitive, B::IntTensorPrimitive) {
        B::int_min_dim_with_indices(tensor, dim)
    }
    fn int_abs(tensor: B::IntTensorPrimitive) -> B::IntTensorPrimitive {
        B::int_abs(tensor)
    }
    fn int_into_float(
        tensor: <Autodiff<B> as Backend>::IntTensorPrimitive,
    ) -> <Autodiff<B> as Backend>::FloatTensorPrimitive {
        AutodiffTensor::new(B::int_into_float(tensor))
    }

    fn int_swap_dims(
        tensor: <Autodiff<B> as Backend>::IntTensorPrimitive,
        dim1: usize,
        dim2: usize,
    ) -> <Autodiff<B> as Backend>::IntTensorPrimitive {
        B::int_swap_dims(tensor, dim1, dim2)
    }

    fn int_narrow(
        tensor: <Autodiff<B> as Backend>::IntTensorPrimitive,
        dim: usize,
        start: usize,
        length: usize,
    ) -> <Autodiff<B> as Backend>::IntTensorPrimitive {
        B::int_narrow(tensor, dim, start, length)
    }

    fn int_chunk(
        tensor: <Autodiff<B> as Backend>::IntTensorPrimitive,
        chunks: usize,
        dim: usize,
    ) -> Vec<<Autodiff<B> as Backend>::IntTensorPrimitive> {
        B::int_chunk(tensor, chunks, dim)
    }

    fn int_split(
        tensor: <Autodiff<B> as Backend>::IntTensorPrimitive,
        split_size: usize,
        dim: usize,
    ) -> Vec<<Autodiff<B> as Backend>::IntTensorPrimitive> {
        B::int_split(tensor, split_size, dim)
    }

    fn int_split_with_sizes(
        tensor: <Autodiff<B> as Backend>::IntTensorPrimitive,
        split_sizes: Vec<usize>,
        dim: usize,
    ) -> Vec<<Autodiff<B> as Backend>::IntTensorPrimitive> {
        B::int_split_with_sizes(tensor, split_sizes, dim)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> IntTensor<Self> {
        B::int_random(shape, distribution, device)
    }

    fn int_arange(range: std::ops::Range<i64>, device: &Device<Self>) -> IntTensor<Self> {
        B::int_arange(range, device)
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        B::int_permute(tensor, axes)
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        B::int_flip(tensor, axes)
    }

    fn int_sign(tensor: IntTensor<Self>) -> IntTensor<Self> {
        B::int_sign(tensor)
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        B::int_prod(tensor)
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        B::int_prod_dim(tensor, dim)
    }

    fn int_expand(tensor: IntTensor<B>, shape: Shape) -> IntTensor<B> {
        B::int_expand(tensor, shape)
    }

    fn int_sort(tensor: IntTensor<Self>, dim: usize, descending: bool) -> IntTensor<Self> {
        B::int_sort(tensor, dim, descending)
    }

    fn int_sort_with_indices(
        tensor: IntTensor<Self>,
        dim: usize,
        descending: bool,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        B::int_sort_with_indices(tensor, dim, descending)
    }

    fn int_argsort(tensor: IntTensor<Self>, dim: usize, descending: bool) -> IntTensor<Self> {
        B::int_argsort(tensor, dim, descending)
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        B::bitwise_and(lhs, rhs)
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: B::IntElem) -> IntTensor<Self> {
        B::bitwise_and_scalar(lhs, rhs)
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        B::bitwise_or(lhs, rhs)
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: B::IntElem) -> IntTensor<Self> {
        B::bitwise_or_scalar(lhs, rhs)
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        B::bitwise_xor(lhs, rhs)
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: B::IntElem) -> IntTensor<Self> {
        B::bitwise_xor_scalar(lhs, rhs)
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        B::bitwise_not(tensor)
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        B::bitwise_left_shift(lhs, rhs)
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: B::IntElem) -> IntTensor<Self> {
        B::bitwise_left_shift_scalar(lhs, rhs)
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        B::bitwise_right_shift(lhs, rhs)
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: B::IntElem) -> IntTensor<Self> {
        B::bitwise_right_shift_scalar(lhs, rhs)
    }
}
