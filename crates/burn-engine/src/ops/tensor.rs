use burn_backend::{
    ExecutionError, Scalar, TensorData,
    ops::FloatTensorOps,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_std::{FloatDType, Shape, Slice};

use crate::backends::*;
use crate::{Device, Engine, EngineTensor};
use crate::{
    binary_float, create_float, dispatch_async_float, multi_tensor_op, to_device, unary_float,
};

// TODO: remove backend default elem type genericsnow that we have per-device defaults
// https://github.com/tracel-ai/burn/issues/3642

impl FloatTensorOps<Self> for Engine {
    fn float_from_data(data: burn_backend::TensorData, device: &Device) -> FloatTensor<Self> {
        create_float!(device, |device| B::float_from_data(data, device))
    }

    fn float_random(
        shape: Shape,
        distribution: burn_backend::Distribution,
        device: &Device,
    ) -> FloatTensor<Self> {
        create_float!(device, |device| B::float_random(
            shape,
            distribution,
            device
        ))
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        dispatch_async_float!(float_into_data, tensor)
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
        unary_float!(float_into_int, tensor => Int)
    }

    fn float_empty(shape: Shape, device: &Device, dtype: FloatDType) -> FloatTensor<Self> {
        create_float!(device, |device| B::float_empty(shape, device, dtype))
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!(float_add, lhs, rhs)
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_float!(float_add_scalar, lhs, rhs)
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!(float_sub, lhs, rhs)
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_float!(float_sub_scalar, lhs, rhs)
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!(float_mul, lhs, rhs)
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_float!(float_mul_scalar, lhs, rhs)
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!(float_div, lhs, rhs)
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_float!(float_div_scalar, lhs, rhs)
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!(float_remainder, lhs, rhs)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        unary_float!(float_remainder_scalar, lhs, rhs)
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!(float_matmul, lhs, rhs)
    }

    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        binary_float!(float_cross, lhs, rhs, dim)
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_recip, tensor)
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        unary_float!(float_swap_dims, tensor, dim1, dim2)
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        unary_float!(float_permute, tensor, axes)
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        unary_float!(float_flip, tensor, axes)
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        unary_float!(float_reshape, tensor, shape)
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        multi_tensor_op!(Float, float(tensor), int(indices), |tensor, indices| {
            B::float_gather(dim, tensor, indices)
        })
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        multi_tensor_op!(
            Float,
            float(tensor),
            int(indices),
            float(value),
            |tensor, indices, value| B::float_scatter_add(dim, tensor, indices, value)
        )
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        multi_tensor_op!(Float, float(tensor), int(indices), |tensor, indices| {
            B::float_select(tensor, dim, indices)
        })
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        multi_tensor_op!(
            Float,
            float(tensor),
            int(indices),
            float(value),
            |tensor, indices, value| B::float_select_add(tensor, dim, indices, value)
        )
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        unary_float!(float_slice, tensor, slices)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        multi_tensor_op!(Float, float(tensor), float(value), |tensor, value| {
            B::float_slice_assign(tensor, slices, value)
        })
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        multi_tensor_op!(
            Float,
            float(tensor),
            bool(mask),
            float(value),
            |tensor, mask, value| B::float_mask_where(tensor, mask, value)
        )
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> FloatTensor<Self> {
        multi_tensor_op!(Float, float(tensor), bool(mask), |tensor, mask| {
            B::float_mask_fill(tensor, mask, value)
        })
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float!(float_equal, lhs, rhs => Bool)
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_float!(float_equal_elem, lhs, rhs => Bool)
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float!(float_greater, lhs, rhs => Bool)
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_float!(float_greater_elem, lhs, rhs => Bool)
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float!(float_greater_equal, lhs, rhs => Bool)
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_float!(float_greater_equal_elem, lhs, rhs => Bool)
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float!(float_lower, lhs, rhs => Bool)
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_float!(float_lower_elem, lhs, rhs => Bool)
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float!(float_lower_equal, lhs, rhs => Bool)
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_float!(float_lower_equal_elem, lhs, rhs => Bool)
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_sum, tensor)
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(float_sum_dim, tensor, dim)
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(float_mean_dim, tensor, dim)
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(float_cumsum, tensor, dim)
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(float_cumprod, tensor, dim)
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(float_cummin, tensor, dim)
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        unary_float!(float_cummax, tensor, dim)
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        unary_float!(float_cast, tensor, dtype)
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_exp, tensor)
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_log, tensor)
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_log1p, tensor)
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!(float_powf, lhs, rhs)
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        unary_float!(float_powf_scalar_impl, tensor, value)
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_sqrt, tensor)
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_abs, tensor)
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_cos, tensor)
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_sin, tensor)
    }

    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_tan, tensor)
    }

    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_cosh, tensor)
    }

    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_sinh, tensor)
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_tanh, tensor)
    }

    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_acos, tensor)
    }

    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_acosh, tensor)
    }

    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_asin, tensor)
    }

    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_asinh, tensor)
    }

    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_atan, tensor)
    }

    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_atanh, tensor)
    }

    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!(float_atan2, lhs, rhs)
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_round, tensor)
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_floor, tensor)
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_ceil, tensor)
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_trunc, tensor)
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(float_erf, tensor)
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_float!(float_argmax, tensor, dim => Int)
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_float!(float_argmin, tensor, dim => Int)
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        unary_float!(float_expand, tensor, shape)
    }

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        unary_float!(float_unfold, tensor, dim, size, step)
    }
}
