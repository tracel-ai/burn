use crate::{checkpoint::strategy::CheckpointStrategy, tensor::AutodiffTensor, Autodiff};

use burn_tensor::{
    backend::Backend,
    ops::{BoolTensor, IntTensor, IntTensorOps},
    Data, Device, Distribution, Reader, Shape,
};

impl<B: Backend, C: CheckpointStrategy> IntTensorOps<Self> for Autodiff<B, C> {
    fn int_from_data<const D: usize>(
        data: Data<B::IntElem, D>,
        device: &Device<Self>,
    ) -> IntTensor<B, D> {
        B::int_from_data(data, device)
    }

    fn int_shape<const D: usize>(tensor: &IntTensor<B, D>) -> Shape<D> {
        B::int_shape(tensor)
    }

    fn int_to_data<const D: usize>(tensor: &IntTensor<B, D>) -> Reader<Data<B::IntElem, D>> {
        B::int_to_data(tensor)
    }

    fn int_into_data<const D: usize>(tensor: IntTensor<B, D>) -> Reader<Data<B::IntElem, D>> {
        B::int_into_data(tensor)
    }

    fn int_to_device<const D: usize>(
        tensor: IntTensor<B, D>,
        device: &Device<Self>,
    ) -> IntTensor<B, D> {
        B::int_to_device(tensor, device)
    }

    fn int_device<const D: usize>(tensor: &IntTensor<B, D>) -> Device<Self> {
        B::int_device(tensor)
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: IntTensor<B, D1>,
        shape: Shape<D2>,
    ) -> IntTensor<B, D2> {
        B::int_reshape(tensor, shape)
    }

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: IntTensor<B, D1>,
        ranges: [std::ops::Range<usize>; D2],
    ) -> IntTensor<B, D1> {
        B::int_slice(tensor, ranges)
    }

    fn int_empty<const D: usize>(
        shape: Shape<D>,
        device: &<Autodiff<B> as Backend>::Device,
    ) -> IntTensor<B, D> {
        B::int_empty(shape, device)
    }

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: IntTensor<B, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: IntTensor<B, D1>,
    ) -> IntTensor<B, D1> {
        B::int_slice_assign(tensor, ranges, value)
    }

    fn int_cat<const D: usize>(tensors: Vec<IntTensor<B, D>>, dim: usize) -> IntTensor<B, D> {
        B::int_cat(tensors, dim)
    }

    fn int_equal<const D: usize>(lhs: IntTensor<B, D>, rhs: IntTensor<B, D>) -> BoolTensor<B, D> {
        B::int_equal(lhs, rhs)
    }

    fn int_equal_elem<const D: usize>(lhs: IntTensor<B, D>, rhs: B::IntElem) -> BoolTensor<B, D> {
        B::int_equal_elem(lhs, rhs)
    }

    fn int_add<const D: usize>(lhs: IntTensor<B, D>, rhs: IntTensor<B, D>) -> IntTensor<B, D> {
        B::int_add(lhs, rhs)
    }

    fn int_add_scalar<const D: usize>(lhs: IntTensor<B, D>, rhs: B::IntElem) -> IntTensor<B, D> {
        B::int_add_scalar(lhs, rhs)
    }

    fn int_clamp_min<const D: usize>(tensor: IntTensor<B, D>, min: B::IntElem) -> IntTensor<B, D> {
        B::int_clamp_min(tensor, min)
    }

    fn int_clamp_max<const D: usize>(tensor: IntTensor<B, D>, max: B::IntElem) -> IntTensor<B, D> {
        B::int_clamp_max(tensor, max)
    }

    fn int_clamp<const D: usize>(
        tensor: IntTensor<B, D>,
        min: B::IntElem,
        max: B::IntElem,
    ) -> IntTensor<B, D> {
        B::int_clamp(tensor, min, max)
    }

    fn int_sub<const D: usize>(lhs: IntTensor<B, D>, rhs: IntTensor<B, D>) -> IntTensor<B, D> {
        B::int_sub(lhs, rhs)
    }

    fn int_sub_scalar<const D: usize>(lhs: IntTensor<B, D>, rhs: B::IntElem) -> IntTensor<B, D> {
        B::int_sub_scalar(lhs, rhs)
    }

    fn int_mul<const D: usize>(lhs: IntTensor<B, D>, rhs: IntTensor<B, D>) -> IntTensor<B, D> {
        B::int_mul(lhs, rhs)
    }

    fn int_mul_scalar<const D: usize>(lhs: IntTensor<B, D>, rhs: B::IntElem) -> IntTensor<B, D> {
        B::int_mul_scalar(lhs, rhs)
    }

    fn int_div<const D: usize>(lhs: IntTensor<B, D>, rhs: IntTensor<B, D>) -> IntTensor<B, D> {
        B::int_div(lhs, rhs)
    }

    fn int_div_scalar<const D: usize>(lhs: IntTensor<B, D>, rhs: B::IntElem) -> IntTensor<B, D> {
        B::int_div_scalar(lhs, rhs)
    }

    fn int_remainder_scalar<const D: usize>(
        lhs: IntTensor<B, D>,
        rhs: B::IntElem,
    ) -> IntTensor<B, D> {
        B::int_remainder_scalar(lhs, rhs)
    }

    fn int_neg<const D: usize>(tensor: IntTensor<B, D>) -> IntTensor<B, D> {
        B::int_neg(tensor)
    }

    fn int_zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<B, D> {
        B::int_zeros(shape, device)
    }

    fn int_ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<B, D> {
        B::int_ones(shape, device)
    }

    fn int_full<const D: usize>(
        shape: Shape<D>,
        fill_value: B::IntElem,
        device: &Device<Self>,
    ) -> IntTensor<B, D> {
        B::int_full(shape, fill_value, device)
    }

    fn int_sum<const D: usize>(tensor: IntTensor<B, D>) -> IntTensor<B, 1> {
        B::int_sum(tensor)
    }

    fn int_sum_dim<const D: usize>(tensor: IntTensor<B, D>, dim: usize) -> IntTensor<B, D> {
        B::int_sum_dim(tensor, dim)
    }

    fn int_mean<const D: usize>(tensor: IntTensor<B, D>) -> IntTensor<B, 1> {
        B::int_mean(tensor)
    }

    fn int_mean_dim<const D: usize>(tensor: IntTensor<B, D>, dim: usize) -> IntTensor<B, D> {
        B::int_mean_dim(tensor, dim)
    }

    fn int_repeat<const D: usize>(
        tensor: IntTensor<B, D>,
        dim: usize,
        times: usize,
    ) -> IntTensor<B, D> {
        B::int_repeat(tensor, dim, times)
    }

    fn int_greater<const D: usize>(lhs: IntTensor<B, D>, rhs: IntTensor<B, D>) -> BoolTensor<B, D> {
        B::int_greater(lhs, rhs)
    }

    fn int_greater_elem<const D: usize>(lhs: IntTensor<B, D>, rhs: B::IntElem) -> BoolTensor<B, D> {
        B::int_greater_elem(lhs, rhs)
    }

    fn int_greater_equal<const D: usize>(
        lhs: IntTensor<B, D>,
        rhs: IntTensor<B, D>,
    ) -> BoolTensor<B, D> {
        B::int_greater_equal(lhs, rhs)
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: IntTensor<B, D>,
        rhs: B::IntElem,
    ) -> BoolTensor<B, D> {
        B::int_greater_equal_elem(lhs, rhs)
    }

    fn int_lower<const D: usize>(lhs: IntTensor<B, D>, rhs: IntTensor<B, D>) -> BoolTensor<B, D> {
        B::int_lower(lhs, rhs)
    }

    fn int_lower_elem<const D: usize>(lhs: IntTensor<B, D>, rhs: B::IntElem) -> BoolTensor<B, D> {
        B::int_lower_elem(lhs, rhs)
    }

    fn int_lower_equal<const D: usize>(
        lhs: IntTensor<B, D>,
        rhs: IntTensor<B, D>,
    ) -> BoolTensor<B, D> {
        B::int_lower_equal(lhs, rhs)
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: IntTensor<B, D>,
        rhs: B::IntElem,
    ) -> BoolTensor<B, D> {
        B::int_lower_equal_elem(lhs, rhs)
    }

    fn int_gather<const D: usize>(
        dim: usize,
        tensor: IntTensor<B, D>,
        indices: IntTensor<B, D>,
    ) -> IntTensor<B, D> {
        B::int_gather(dim, tensor, indices)
    }

    fn int_scatter<const D: usize>(
        dim: usize,
        tensor: IntTensor<B, D>,
        indices: IntTensor<B, D>,
        value: IntTensor<B, D>,
    ) -> IntTensor<B, D> {
        B::int_scatter(dim, tensor, indices, value)
    }

    fn int_select<const D: usize>(
        tensor: IntTensor<B, D>,
        dim: usize,
        indices: IntTensor<B, 1>,
    ) -> IntTensor<B, D> {
        B::int_select(tensor, dim, indices)
    }

    fn int_select_assign<const D: usize>(
        tensor: IntTensor<B, D>,
        dim: usize,
        indices: IntTensor<B, 1>,
        value: IntTensor<B, D>,
    ) -> IntTensor<B, D> {
        B::int_select_assign(tensor, dim, indices, value)
    }

    fn int_mask_where<const D: usize>(
        tensor: IntTensor<B, D>,
        mask: BoolTensor<B, D>,
        value: IntTensor<B, D>,
    ) -> <Autodiff<B> as Backend>::IntTensorPrimitive<D> {
        B::int_mask_where(tensor, mask, value)
    }

    fn int_mask_fill<const D: usize>(
        tensor: IntTensor<B, D>,
        mask: BoolTensor<B, D>,
        value: B::IntElem,
    ) -> <Autodiff<B> as Backend>::IntTensorPrimitive<D> {
        B::int_mask_fill(tensor, mask, value)
    }

    fn int_argmax<const D: usize>(tensor: IntTensor<B, D>, dim: usize) -> IntTensor<B, D> {
        B::int_argmax(tensor, dim)
    }
    fn int_argmin<const D: usize>(tensor: IntTensor<B, D>, dim: usize) -> IntTensor<B, D> {
        B::int_argmin(tensor, dim)
    }
    fn int_max<const D: usize>(tensor: B::IntTensorPrimitive<D>) -> B::IntTensorPrimitive<1> {
        B::int_max(tensor)
    }
    fn int_max_dim<const D: usize>(
        tensor: B::IntTensorPrimitive<D>,
        dim: usize,
    ) -> B::IntTensorPrimitive<D> {
        B::int_max_dim(tensor, dim)
    }
    fn int_max_dim_with_indices<const D: usize>(
        tensor: B::IntTensorPrimitive<D>,
        dim: usize,
    ) -> (B::IntTensorPrimitive<D>, B::IntTensorPrimitive<D>) {
        B::int_max_dim_with_indices(tensor, dim)
    }
    fn int_min<const D: usize>(tensor: B::IntTensorPrimitive<D>) -> B::IntTensorPrimitive<1> {
        B::int_min(tensor)
    }
    fn int_min_dim<const D: usize>(
        tensor: B::IntTensorPrimitive<D>,
        dim: usize,
    ) -> B::IntTensorPrimitive<D> {
        B::int_min_dim(tensor, dim)
    }
    fn int_min_dim_with_indices<const D: usize>(
        tensor: B::IntTensorPrimitive<D>,
        dim: usize,
    ) -> (B::IntTensorPrimitive<D>, B::IntTensorPrimitive<D>) {
        B::int_min_dim_with_indices(tensor, dim)
    }
    fn int_abs<const D: usize>(tensor: B::IntTensorPrimitive<D>) -> B::IntTensorPrimitive<D> {
        B::int_abs(tensor)
    }
    fn int_into_float<const D: usize>(
        tensor: <Autodiff<B> as Backend>::IntTensorPrimitive<D>,
    ) -> <Autodiff<B> as Backend>::FloatTensorPrimitive<D> {
        AutodiffTensor::new(B::int_into_float(tensor))
    }

    fn int_swap_dims<const D: usize>(
        tensor: <Autodiff<B> as Backend>::IntTensorPrimitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> <Autodiff<B> as Backend>::IntTensorPrimitive<D> {
        B::int_swap_dims(tensor, dim1, dim2)
    }

    fn int_narrow<const D: usize>(
        tensor: <Autodiff<B> as Backend>::IntTensorPrimitive<D>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> <Autodiff<B> as Backend>::IntTensorPrimitive<D> {
        B::int_narrow(tensor, dim, start, length)
    }

    fn int_chunk<const D: usize>(
        tensor: <Autodiff<B> as Backend>::IntTensorPrimitive<D>,
        chunks: usize,
        dim: usize,
    ) -> Vec<<Autodiff<B> as Backend>::IntTensorPrimitive<D>> {
        B::int_chunk(tensor, chunks, dim)
    }

    fn int_random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        B::int_random(shape, distribution, device)
    }

    fn int_arange(range: std::ops::Range<i64>, device: &Device<Self>) -> IntTensor<Self, 1> {
        B::int_arange(range, device)
    }

    fn int_permute<const D: usize>(
        tensor: IntTensor<Self, D>,
        axes: [usize; D],
    ) -> IntTensor<Self, D> {
        B::int_permute(tensor, axes)
    }

    fn int_flip<const D: usize>(tensor: IntTensor<Self, D>, axes: &[usize]) -> IntTensor<Self, D> {
        B::int_flip(tensor, axes)
    }

    fn int_sign<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, D> {
        B::int_sign(tensor)
    }

    fn int_prod<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        B::int_prod(tensor)
    }

    fn int_prod_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        B::int_prod_dim(tensor, dim)
    }

    fn int_expand<const D: usize, const D2: usize>(
        tensor: IntTensor<B, D>,
        shape: Shape<D2>,
    ) -> IntTensor<B, D2> {
        B::int_expand(tensor, shape)
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn int_sort<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        descending: bool,
    ) -> IntTensor<Self, D> {
        B::int_sort(tensor, dim, descending)
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn int_sort_with_indices<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        descending: bool,
    ) -> (IntTensor<Self, D>, IntTensor<Self, D>) {
        B::int_sort_with_indices(tensor, dim, descending)
    }

    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn int_argsort<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        descending: bool,
    ) -> IntTensor<Self, D> {
        B::int_argsort(tensor, dim, descending)
    }
}
