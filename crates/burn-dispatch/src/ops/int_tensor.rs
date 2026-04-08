use alloc::vec::Vec;
use burn_backend::{
    BoolDType, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
    ops::IntTensorOps,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};

use crate::backends::*;
use crate::{Dispatch, DispatchDevice};

impl IntTensorOps<Self> for Dispatch {
    fn int_empty(shape: Shape, device: &DispatchDevice, dtype: IntDType) -> IntTensor<Self> {
        creation_op!(Int, device, |device| B::int_empty(shape, device, dtype))
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> Result<TensorData, ExecutionError> {
        unary_op!(tensor, int, |tensor| B::int_into_data(tensor).await)
    }

    fn int_from_data(data: TensorData, device: &DispatchDevice) -> IntTensor<Self> {
        creation_op!(Int, device, |device| B::int_from_data(data, device))
    }

    fn int_device(tensor: &IntTensor<Self>) -> DispatchDevice {
        tensor.device()
    }

    fn int_to_device(tensor: IntTensor<Self>, device: &DispatchDevice) -> IntTensor<Self> {
        to_device!(Int, int, tensor, device, int_to_device, |inner, device| {
            let data = burn_backend::read_sync(B1::int_into_data(inner)).expect("Should read data");
            B2::int_from_data(data, device)
        })
    }

    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_reshape(tensor, shape) => Int)
    }

    fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_slice(tensor, slices) => Int)
    }

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        slices: &[Slice],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        binary_op!((tensor, int), (value, int), |tensor, value| B::int_slice_assign(tensor, slices, value) => Int)
    }

    fn int_into_float(tensor: IntTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_into_float(tensor, out_dtype) => Float)
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        multi_op!(
            inputs[(tensor, int), (mask, bool), (value, int)], => Int,
            B::int_mask_where(tensor, mask, value)
        )
    }

    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> IntTensor<Self> {
        binary_op!((tensor, int), (mask, bool), |tensor, mask| B::int_mask_fill(tensor, mask, value) => Int)
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        binary_op!((tensor, int), (indices, int), |tensor, indices| B::int_gather(dim, tensor, indices) => Int)
    }

    fn int_scatter_add(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        multi_op!(
            inputs[(tensor, int), (indices, int), (value, int)], => Int,
            B::int_scatter_add(dim, tensor, indices, value)
        )
    }

    fn int_scatter_nd(
        data: IntTensor<Self>,
        indices: IntTensor<Self>,
        values: IntTensor<Self>,
        reduction: burn_backend::tensor::IndexingUpdateOp,
    ) -> IntTensor<Self> {
        multi_op!(
            inputs[(data, int), (indices, int), (values, int)], => Int,
            B::int_scatter_nd(data, indices, values, reduction)
        )
    }

    fn int_gather_nd(data: IntTensor<Self>, indices: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((data, int), (indices, int), |data, indices| B::int_gather_nd(data, indices) => Int)
    }

    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        binary_op!((tensor, int), (indices, int), |tensor, indices| B::int_select(tensor, dim, indices) => Int)
    }

    fn int_select_add(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        multi_op!(
            inputs[(tensor, int), (indices, int), (value, int)], => Int,
            B::int_select_add(tensor, dim, indices, value)
        )
    }

    fn int_equal(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_equal(lhs, rhs, out_dtype) => Bool)
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        unary_op!(lhs, int, |lhs| B::int_equal_elem(lhs, rhs, out_dtype) => Bool)
    }

    fn int_greater(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_greater(lhs, rhs, out_dtype) => Bool)
    }

    fn int_greater_elem(
        lhs: IntTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unary_op!(lhs, int, |lhs| B::int_greater_elem(lhs, rhs, out_dtype) => Bool)
    }

    fn int_greater_equal(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_greater_equal(lhs, rhs, out_dtype) => Bool)
    }

    fn int_greater_equal_elem(
        lhs: IntTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unary_op!(lhs, int, |lhs| B::int_greater_equal_elem(lhs, rhs, out_dtype) => Bool)
    }

    fn int_lower(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_lower(lhs, rhs, out_dtype) => Bool)
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<Self> {
        unary_op!(lhs, int, |lhs| B::int_lower_elem(lhs, rhs, out_dtype) => Bool)
    }

    fn int_lower_equal(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_lower_equal(lhs, rhs, out_dtype) => Bool)
    }

    fn int_lower_equal_elem(
        lhs: IntTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unary_op!(lhs, int, |lhs| B::int_lower_equal_elem(lhs, rhs, out_dtype) => Bool)
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_add(lhs, rhs) => Int)
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, int, |lhs| B::int_add_scalar(lhs, rhs) => Int)
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_sub(lhs, rhs) => Int)
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, int, |lhs| B::int_sub_scalar(lhs, rhs) => Int)
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_mul(lhs, rhs) => Int)
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, int, |lhs| B::int_mul_scalar(lhs, rhs) => Int)
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_div(lhs, rhs) => Int)
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, int, |lhs| B::int_div_scalar(lhs, rhs) => Int)
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_remainder(lhs, rhs) => Int)
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, int, |lhs| B::int_remainder_scalar(lhs, rhs) => Int)
    }

    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_matmul(lhs, rhs) => Int)
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_sum(tensor) => Int)
    }

    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_sum_dim(tensor, dim) => Int)
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_prod(tensor) => Int)
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_prod_dim(tensor, dim) => Int)
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_mean_dim(tensor, dim) => Int)
    }

    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_cumsum(tensor, dim) => Int)
    }

    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_cumprod(tensor, dim) => Int)
    }

    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_cummin(tensor, dim) => Int)
    }

    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_cummax(tensor, dim) => Int)
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_argmax(tensor, dim) => Int)
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_argmin(tensor, dim) => Int)
    }

    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_abs(tensor) => Int)
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_swap_dims(tensor, dim1, dim2) => Int)
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_permute(tensor, axes) => Int)
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_flip(tensor, axes) => Int)
    }

    fn int_random(
        shape: Shape,
        distribution: burn_backend::Distribution,
        device: &DispatchDevice,
        dtype: IntDType,
    ) -> IntTensor<Self> {
        creation_op!(Int, device, |device| {
            B::int_random(shape, distribution, device, dtype)
        })
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_expand(tensor, shape) => Int)
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::bitwise_and(lhs, rhs) => Int)
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, int, |lhs| B::bitwise_and_scalar(lhs, rhs) => Int)
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::bitwise_or(lhs, rhs) => Int)
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, int, |lhs| B::bitwise_or_scalar(lhs, rhs) => Int)
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::bitwise_xor(lhs, rhs) => Int)
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, int, |lhs| B::bitwise_xor_scalar(lhs, rhs) => Int)
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::bitwise_not(tensor) => Int)
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::bitwise_left_shift(lhs, rhs) => Int)
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, int, |lhs| B::bitwise_left_shift_scalar(lhs, rhs) => Int)
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::bitwise_right_shift(lhs, rhs) => Int)
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, int, |lhs| B::bitwise_right_shift_scalar(lhs, rhs) => Int)
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_cast(tensor, dtype) => Int)
    }

    fn int_unfold(
        tensor: IntTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_unfold(tensor, dim, size, step) => Int)
    }

    fn int_repeat_dim(tensor: IntTensor<Self>, dim: usize, times: usize) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_repeat_dim(tensor, dim, times) => Int)
    }

    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        vec_op!(tensors, int, |tensors| B::int_cat(tensors, dim) => Int)
    }

    fn int_not_equal(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_not_equal(lhs, rhs, out_dtype) => Bool)
    }

    fn int_not_equal_elem(
        lhs: IntTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unary_op!(lhs, int, |lhs| B::int_not_equal_elem(lhs, rhs, out_dtype) => Bool)
    }

    fn int_powi(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_powi(lhs, rhs) => Int)
    }

    fn int_powi_scalar_impl(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, int, |lhs| B::int_powi_scalar_impl(lhs, rhs) => Int)
    }

    fn int_clamp_min(tensor: IntTensor<Self>, min: Scalar) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_clamp_min(tensor, min) => Int)
    }

    fn int_clamp_max(tensor: IntTensor<Self>, max: Scalar) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_clamp_max(tensor, max) => Int)
    }

    fn int_clamp(tensor: IntTensor<Self>, min: Scalar, max: Scalar) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_clamp(tensor, min, max) => Int)
    }

    fn int_neg(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_neg(tensor) => Int)
    }

    fn int_zeros(shape: Shape, device: &DispatchDevice, dtype: IntDType) -> IntTensor<Self> {
        creation_op!(Int, device, |device| B::int_zeros(shape, device, dtype))
    }

    fn int_ones(shape: Shape, device: &DispatchDevice, dtype: IntDType) -> IntTensor<Self> {
        creation_op!(Int, device, |device| B::int_ones(shape, device, dtype))
    }

    fn int_full(
        shape: Shape,
        fill_value: Scalar,
        device: &DispatchDevice,
        dtype: IntDType,
    ) -> IntTensor<Self> {
        creation_op!(Int, device, |device| B::int_full(
            shape, fill_value, device, dtype
        ))
    }

    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_mean(tensor) => Int)
    }

    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_max(tensor) => Int)
    }

    fn int_max_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_max_dim(tensor, dim) => Int)
    }

    fn int_max_dim_with_indices(
        tensor: IntTensor<Self>,
        dim: usize,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        multi_op!(
            inputs[(tensor, int)],
            outputs[(out, Int), (indices, Int)],
            B::int_max_dim_with_indices(tensor, dim)
        )
    }

    fn int_max_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_max_abs(tensor) => Int)
    }

    fn int_max_abs_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_max_abs_dim(tensor, dim) => Int)
    }

    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_min(tensor) => Int)
    }

    fn int_min_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_min_dim(tensor, dim) => Int)
    }

    fn int_min_dim_with_indices(
        tensor: IntTensor<Self>,
        dim: usize,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        multi_op!(
            inputs[(tensor, int)],
            outputs[(out, Int), (indices, Int)],
            B::int_min_dim_with_indices(tensor, dim)
        )
    }

    fn int_transpose(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_transpose(tensor) => Int)
    }

    fn int_arange_step(
        range: core::ops::Range<i64>,
        step: usize,
        device: &DispatchDevice,
        dtype: IntDType,
    ) -> IntTensor<Self> {
        creation_op!(Int, device, |device| B::int_arange_step(
            range, step, device, dtype
        ))
    }

    fn int_arange(
        range: core::ops::Range<i64>,
        device: &DispatchDevice,
        dtype: IntDType,
    ) -> IntTensor<Self> {
        creation_op!(Int, device, |device| B::int_arange(range, device, dtype))
    }

    fn int_any(tensor: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_any(tensor, out_dtype) => Bool)
    }

    fn int_any_dim(tensor: IntTensor<Self>, dim: usize, out_dtype: BoolDType) -> BoolTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_any_dim(tensor, dim, out_dtype) => Bool)
    }

    fn int_all(tensor: IntTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_all(tensor, out_dtype) => Bool)
    }

    fn int_all_dim(tensor: IntTensor<Self>, dim: usize, out_dtype: BoolDType) -> BoolTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_all_dim(tensor, dim, out_dtype) => Bool)
    }

    fn int_sign(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_sign(tensor) => Int)
    }

    fn int_sort(tensor: IntTensor<Self>, dim: usize, descending: bool) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_sort(tensor, dim, descending) => Int)
    }

    fn int_sort_with_indices(
        tensor: IntTensor<Self>,
        dim: usize,
        descending: bool,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        multi_op!(
            inputs[(tensor, int)],
            outputs[(out, Int), (indices, Int)],
            B::int_sort_with_indices(tensor, dim, descending)
        )
    }

    fn int_argsort(tensor: IntTensor<Self>, dim: usize, descending: bool) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_argsort(tensor, dim, descending) => Int)
    }
}
