use burn_backend::{
    ExecutionError, Scalar, TensorData,
    ops::FloatTensorOps,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_std::{FloatDType, Shape, Slice};

use crate::backends::*;
use crate::{Device, Dispatch};

// TODO: remove backend default elem type genericsnow that we have per-device defaults
// https://github.com/tracel-ai/burn/issues/3642

impl FloatTensorOps<Self> for Dispatch {
    fn float_from_data(data: burn_backend::TensorData, device: &Device) -> FloatTensor<Self> {
        creation_op!(Float, device, |device| B::float_from_data(data, device))
    }

    fn float_random(
        shape: Shape,
        distribution: burn_backend::Distribution,
        device: &Device,
    ) -> FloatTensor<Self> {
        creation_op!(Float, device, |device| {
            B::float_random(shape, distribution, device)
        })
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        unary_float!(tensor, float, |tensor| B::float_into_data(tensor).await)
    }

    fn float_device(tensor: &FloatTensor<Self>) -> Device {
        tensor.device()
    }

    fn float_to_device(tensor: FloatTensor<Self>, device: &Device) -> FloatTensor<Self> {
        float_to_device!(
            Float,
            float,
            tensor,
            device,
            float_to_device,
            |inner, device| {
                let data =
                    burn_backend::read_sync(B1::float_into_data(inner)).expect("Should read data");
                B2::float_from_data(data, device)
            }
        )
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> IntTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_into_int(tensor) => Int)
    }

    fn float_empty(shape: Shape, device: &Device, dtype: FloatDType) -> FloatTensor<Self> {
        creation_op!(Float, device, |device| B::float_empty(shape, device, dtype))
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!((lhs, float), (rhs, float), |lhs, rhs| B::float_add(lhs, rhs) => Float)
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_float!(lhs, float, |lhs| B::float_add_scalar(lhs, rhs) => Float)
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!((lhs, float), (rhs, float), |lhs, rhs| B::float_sub(lhs, rhs) => Float)
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_float!(lhs, float, |lhs| B::float_sub_scalar(lhs, rhs) => Float)
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!((lhs, float), (rhs, float), |lhs, rhs| B::float_mul(lhs, rhs) => Float)
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_float!(lhs, float, |lhs| B::float_mul_scalar(lhs, rhs) => Float)
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!((lhs, float), (rhs, float), |lhs, rhs| B::float_div(lhs, rhs) => Float)
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_float!(lhs, float, |lhs| B::float_div_scalar(lhs, rhs) => Float)
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!((lhs, float), (rhs, float), |lhs, rhs| B::float_remainder(lhs, rhs) => Float)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_float!(lhs, float, |lhs| B::float_remainder_scalar(lhs, rhs) => Float)
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!((lhs, float), (rhs, float), |lhs, rhs| B::float_matmul(lhs, rhs) => Float)
    }

    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        binary_float!((lhs, float), (rhs, float), |lhs, rhs| B::float_cross(lhs, rhs, dim) => Float)
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_recip(tensor) => Float)
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_swap_dims(tensor, dim1, dim2) => Float)
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_permute(tensor, axes) => Float)
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_flip(tensor, axes) => Float)
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_reshape(tensor, shape) => Float)
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        binary_float!((tensor, float), (indices, int), |tensor, indices| B::float_gather(dim, tensor, indices) => Float)
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        multi_op!(
            inputs[(tensor, float), (indices, int), (value, float)], => Float,
            B::float_scatter_add(dim, tensor, indices, value)
        )
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        binary_float!((tensor, float), (indices, int), |tensor, indices| B::float_select(tensor, dim, indices) => Float)
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        multi_op!(
            inputs[(tensor, float), (indices, int), (value, float)], => Float,
            B::float_select_add(tensor, dim, indices, value)
        )
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_slice(tensor, slices) => Float)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        binary_float!((tensor, float), (value, float), |tensor, value| B::float_slice_assign(tensor, slices, value) => Float)
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        multi_op!(
            inputs[(tensor, float), (mask, bool), (value, float)], => Float,
            B::float_mask_where(tensor, mask, value)
        )
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> FloatTensor<Self> {
        binary_float!((tensor, float), (mask, bool), |tensor, mask| B::float_mask_fill(tensor, mask, value) => Float)
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float!((lhs, float), (rhs, float), |lhs, rhs| B::float_equal(lhs, rhs) => Bool)
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_float!(lhs, float, |lhs| B::float_equal_elem(lhs, rhs) => Bool)
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float!((lhs, float), (rhs, float), |lhs, rhs| B::float_greater(lhs, rhs) => Bool)
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_float!(lhs, float, |lhs| B::float_greater_elem(lhs, rhs) => Bool)
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float!((lhs, float), (rhs, float), |lhs, rhs| B::float_greater_equal(lhs, rhs) => Bool)
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_float!(lhs, float, |lhs| B::float_greater_equal_elem(lhs, rhs) => Bool)
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float!((lhs, float), (rhs, float), |lhs, rhs| B::float_lower(lhs, rhs) => Bool)
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_float!(lhs, float, |lhs| B::float_lower_elem(lhs, rhs) => Bool)
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float!((lhs, float), (rhs, float), |lhs, rhs| B::float_lower_equal(lhs, rhs) => Bool)
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_float!(lhs, float, |lhs| B::float_lower_equal_elem(lhs, rhs) => Bool)
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_sum(tensor) => Float)
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_sum_dim(tensor, dim) => Float)
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_mean_dim(tensor, dim) => Float)
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_cumsum(tensor, dim) => Float)
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_cumprod(tensor, dim) => Float)
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_cummin(tensor, dim) => Float)
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_cummax(tensor, dim) => Float)
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_cast(tensor, dtype) => Float)
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_exp(tensor) => Float)
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_log(tensor) => Float)
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_log1p(tensor) => Float)
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!((lhs, float), (rhs, float), |lhs, rhs| B::float_powf(lhs, rhs) => Float)
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_powf_scalar_impl(tensor, value) => Float)
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_sqrt(tensor) => Float)
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_abs(tensor) => Float)
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_cos(tensor) => Float)
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_sin(tensor) => Float)
    }

    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_tan(tensor) => Float)
    }

    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_cosh(tensor) => Float)
    }

    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_sinh(tensor) => Float)
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_tanh(tensor) => Float)
    }

    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_acos(tensor) => Float)
    }

    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_acosh(tensor) => Float)
    }

    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_asin(tensor) => Float)
    }

    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_asinh(tensor) => Float)
    }

    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_atan(tensor) => Float)
    }

    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_atanh(tensor) => Float)
    }

    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!((lhs, float), (rhs, float), |lhs, rhs| B::float_atan2(lhs, rhs) => Float)
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_round(tensor) => Float)
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_floor(tensor) => Float)
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_ceil(tensor) => Float)
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_trunc(tensor) => Float)
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_erf(tensor) => Float)
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_argmax(tensor, dim) => Int)
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_argmin(tensor, dim) => Int)
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_expand(tensor, shape) => Float)
    }

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| {
            B::float_unfold(tensor, dim, size, step)
        } => Float)
    }

    fn float_detach(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_detach(tensor) => Float)
    }

    fn float_set_require_grad(tensor: FloatTensor<Self>, require_grad: bool) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_set_require_grad(tensor, require_grad) => Float)
    }

    fn float_is_require_grad(tensor: &FloatTensor<Self>) -> bool {
        unary_float!(ref tensor, float, |tensor| B::float_is_require_grad(tensor))
    }

    // Default implementation
    fn float_zeros(shape: Shape, device: &Device, dtype: FloatDType) -> FloatTensor<Self> {
        creation_op!(Float, device, |device| B::float_zeros(shape, device, dtype))
    }

    fn float_ones(shape: Shape, device: &Device, dtype: FloatDType) -> FloatTensor<Self> {
        creation_op!(Float, device, |device| B::float_ones(shape, device, dtype))
    }

    fn float_full(
        shape: Shape,
        fill_value: Scalar,
        device: &Device,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        creation_op!(Float, device, |device| B::float_full(
            shape, fill_value, device, dtype
        ))
    }

    fn float_repeat_dim(tensor: FloatTensor<Self>, dim: usize, times: usize) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_repeat_dim(tensor, dim, times) => Float)
    }

    fn float_clamp_min(tensor: FloatTensor<Self>, min: Scalar) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_clamp_min(tensor, min) => Float)
    }

    fn float_clamp_max(tensor: FloatTensor<Self>, max: Scalar) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_clamp_max(tensor, max) => Float)
    }

    fn float_clamp(tensor: FloatTensor<Self>, min: Scalar, max: Scalar) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_clamp(tensor, min, max) => Float)
    }

    fn float_neg(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_neg(tensor) => Float)
    }

    fn float_transpose(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_transpose(tensor) => Float)
    }

    fn float_not_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float!((lhs, float), (rhs, float), |lhs, rhs| B::float_not_equal(lhs, rhs) => Bool)
    }

    fn float_not_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_float!(lhs, float, |lhs| B::float_not_equal_elem(lhs, rhs) => Bool)
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_prod(tensor) => Float)
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_prod_dim(tensor, dim) => Float)
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_mean(tensor) => Float)
    }

    fn float_powi(lhs: FloatTensor<Self>, rhs: IntTensor<Self>) -> FloatTensor<Self> {
        binary_float!((lhs, float), (rhs, int), |lhs, rhs| B::float_powi(lhs, rhs) => Float)
    }

    fn float_powi_scalar_impl(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_float!(lhs, float, |lhs| B::float_powi_scalar_impl(lhs, rhs) => Float)
    }

    fn float_powf_scalar(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_powf_scalar(tensor, value) => Float)
    }

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        vec_op!(tensors, float, |tensors| B::float_cat(tensors, dim) => Float)
    }

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_max(tensor) => Float)
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_max_dim(tensor, dim) => Float)
    }

    fn float_max_dim_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        multi_op!(
            inputs[(tensor, float)],
            outputs[(out, Float), (indices, Int)],
            B::float_max_dim_with_indices(tensor, dim)
        )
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_min(tensor) => Float)
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_min_dim(tensor, dim) => Float)
    }

    fn float_min_dim_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        multi_op!(
            inputs[(tensor, float)],
            outputs[(out, Float), (indices, Int)],
            B::float_min_dim_with_indices(tensor, dim)
        )
    }

    fn float_max_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_max_abs(tensor) => Float)
    }

    fn float_max_abs_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_max_abs_dim(tensor, dim) => Float)
    }

    fn float_any(tensor: FloatTensor<Self>) -> BoolTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_any(tensor) => Bool)
    }

    fn float_any_dim(tensor: FloatTensor<Self>, dim: usize) -> BoolTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_any_dim(tensor, dim) => Bool)
    }

    fn float_all(tensor: FloatTensor<Self>) -> BoolTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_all(tensor) => Bool)
    }

    fn float_all_dim(tensor: FloatTensor<Self>, dim: usize) -> BoolTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_all_dim(tensor, dim) => Bool)
    }

    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_sign(tensor) => Float)
    }

    fn float_sort(tensor: FloatTensor<Self>, dim: usize, descending: bool) -> FloatTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_sort(tensor, dim, descending) => Float)
    }

    fn float_sort_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
        descending: bool,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        multi_op!(
            inputs[(tensor, float)],
            outputs[(out, Float), (indices, Int)],
            B::float_sort_with_indices(tensor, dim, descending)
        )
    }

    fn float_argsort(tensor: FloatTensor<Self>, dim: usize, descending: bool) -> IntTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_argsort(tensor, dim, descending) => Int)
    }

    fn float_grid_sample_2d(
        tensor: FloatTensor<Self>,
        grid: FloatTensor<Self>,
        options: burn_backend::ops::GridSampleOptions,
    ) -> FloatTensor<Self> {
        binary_float!((tensor, float), (grid, float), |tensor, grid| B::float_grid_sample_2d(tensor, grid, options) => Float)
    }

    fn float_is_nan(tensor: FloatTensor<Self>) -> BoolTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_is_nan(tensor) => Bool)
    }

    fn float_is_inf(tensor: FloatTensor<Self>) -> BoolTensor<Self> {
        unary_float!(tensor, float, |tensor| B::float_is_inf(tensor) => Bool)
    }
}
