use burn_backend::{
    ExecutionError, Scalar, TensorData,
    ops::FloatTensorOps,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_std::{FloatDType, Shape, Slice};

use crate::backends::*;
use crate::{Device, Engine};
use crate::{binary_op, creation_op, multi_tensor_op, to_device, unary_op};

// TODO: remove backend default elem type genericsnow that we have per-device defaults
// https://github.com/tracel-ai/burn/issues/3642

impl FloatTensorOps<Self> for Engine {
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
        unary_op!(tensor, float, |tensor| B::float_into_data(tensor).await)
    }

    fn float_device(tensor: &FloatTensor<Self>) -> Device {
        tensor.device()
    }

    fn float_to_device(tensor: FloatTensor<Self>, device: &Device) -> FloatTensor<Self> {
        to_device!(
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
        unary_op!(tensor, float, |tensor| B::float_into_int(tensor) => Int)
    }

    fn float_empty(shape: Shape, device: &Device, dtype: FloatDType) -> FloatTensor<Self> {
        creation_op!(Float, device, |device| B::float_empty(shape, device, dtype))
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_op!((lhs, float), (rhs, float), |lhs, rhs| B::float_add(lhs, rhs) => Float)
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_op!(lhs, float, |lhs| B::float_add_scalar(lhs, rhs) => Float)
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_op!((lhs, float), (rhs, float), |lhs, rhs| B::float_sub(lhs, rhs) => Float)
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_op!(lhs, float, |lhs| B::float_sub_scalar(lhs, rhs) => Float)
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_op!((lhs, float), (rhs, float), |lhs, rhs| B::float_mul(lhs, rhs) => Float)
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_op!(lhs, float, |lhs| B::float_mul_scalar(lhs, rhs) => Float)
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_op!((lhs, float), (rhs, float), |lhs, rhs| B::float_div(lhs, rhs) => Float)
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_op!(lhs, float, |lhs| B::float_div_scalar(lhs, rhs) => Float)
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_op!((lhs, float), (rhs, float), |lhs, rhs| B::float_remainder(lhs, rhs) => Float)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_op!(lhs, float, |lhs| B::float_remainder_scalar(lhs, rhs) => Float)
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_op!((lhs, float), (rhs, float), |lhs, rhs| B::float_matmul(lhs, rhs) => Float)
    }

    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        binary_op!((lhs, float), (rhs, float), |lhs, rhs| B::float_cross(lhs, rhs, dim) => Float)
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_recip(tensor) => Float)
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_swap_dims(tensor, dim1, dim2) => Float)
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_permute(tensor, axes) => Float)
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_flip(tensor, axes) => Float)
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_reshape(tensor, shape) => Float)
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        binary_op!((tensor, float), (indices, int), |tensor, indices| B::float_gather(dim, tensor, indices) => Float)
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        multi_tensor_op!(
            (tensor, float),
            (indices, int),
            (value, float),
            |tensor, indices, value| B::float_scatter_add(dim, tensor, indices, value) => Float
        )
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        binary_op!((tensor, float), (indices, int), |tensor, indices| B::float_select(tensor, dim, indices) => Float)
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        multi_tensor_op!(
            (tensor, float),
            (indices, int),
            (value, float),
            |tensor, indices, value| B::float_select_add(tensor, dim, indices, value) => Float
        )
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_slice(tensor, slices) => Float)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        binary_op!((tensor, float), (value, float), |tensor, value| B::float_slice_assign(tensor, slices, value) => Float)
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        multi_tensor_op!(
            (tensor, float),
            (mask, bool),
            (value, float),
            |tensor, mask, value| B::float_mask_where(tensor, mask, value) => Float
        )
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> FloatTensor<Self> {
        binary_op!((tensor, float), (mask, bool), |tensor, mask| B::float_mask_fill(tensor, mask, value) => Float)
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, float), (rhs, float), |lhs, rhs| B::float_equal(lhs, rhs) => Bool)
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_op!(lhs, float, |lhs| B::float_equal_elem(lhs, rhs) => Bool)
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, float), (rhs, float), |lhs, rhs| B::float_greater(lhs, rhs) => Bool)
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_op!(lhs, float, |lhs| B::float_greater_elem(lhs, rhs) => Bool)
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, float), (rhs, float), |lhs, rhs| B::float_greater_equal(lhs, rhs) => Bool)
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_op!(lhs, float, |lhs| B::float_greater_equal_elem(lhs, rhs) => Bool)
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, float), (rhs, float), |lhs, rhs| B::float_lower(lhs, rhs) => Bool)
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_op!(lhs, float, |lhs| B::float_lower_elem(lhs, rhs) => Bool)
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, float), (rhs, float), |lhs, rhs| B::float_lower_equal(lhs, rhs) => Bool)
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_op!(lhs, float, |lhs| B::float_lower_equal_elem(lhs, rhs) => Bool)
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_sum(tensor) => Float)
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_sum_dim(tensor, dim) => Float)
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_mean_dim(tensor, dim) => Float)
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_cumsum(tensor, dim) => Float)
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_cumprod(tensor, dim) => Float)
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_cummin(tensor, dim) => Float)
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_cummax(tensor, dim) => Float)
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_cast(tensor, dtype) => Float)
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_exp(tensor) => Float)
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_log(tensor) => Float)
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_log1p(tensor) => Float)
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_op!((lhs, float), (rhs, float), |lhs, rhs| B::float_powf(lhs, rhs) => Float)
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_powf_scalar_impl(tensor, value) => Float)
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_sqrt(tensor) => Float)
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_abs(tensor) => Float)
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_cos(tensor) => Float)
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_sin(tensor) => Float)
    }

    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_tan(tensor) => Float)
    }

    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_cosh(tensor) => Float)
    }

    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_sinh(tensor) => Float)
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_tanh(tensor) => Float)
    }

    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_acos(tensor) => Float)
    }

    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_acosh(tensor) => Float)
    }

    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_asin(tensor) => Float)
    }

    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_asinh(tensor) => Float)
    }

    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_atan(tensor) => Float)
    }

    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_atanh(tensor) => Float)
    }

    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_op!((lhs, float), (rhs, float), |lhs, rhs| B::float_atan2(lhs, rhs) => Float)
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_round(tensor) => Float)
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_floor(tensor) => Float)
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_ceil(tensor) => Float)
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_trunc(tensor) => Float)
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_erf(tensor) => Float)
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_argmax(tensor, dim) => Int)
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_argmin(tensor, dim) => Int)
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::float_expand(tensor, shape) => Float)
    }

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| {
            B::float_unfold(tensor, dim, size, step)
        } => Float)
    }
}
