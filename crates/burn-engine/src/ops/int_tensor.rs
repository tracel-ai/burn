use burn_backend::{
    ExecutionError, Scalar, TensorData,
    ops::IntTensorOps,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_std::{IntDType, Shape, Slice};

use crate::backends::*;
use crate::{Device, Engine};
use crate::{binary_op, creation_op, multi_tensor_op, to_device, unary_op};

impl IntTensorOps<Self> for Engine {
    fn int_empty(shape: Shape, device: &Device, dtype: IntDType) -> IntTensor<Self> {
        creation_op!(Int, device, |device| B::int_empty(shape, device, dtype))
    }

    async fn int_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        unary_op!(tensor, float, |tensor| B::int_into_data(tensor).await)
    }

    fn int_from_data(data: TensorData, device: &Device) -> IntTensor<Self> {
        creation_op!(Int, device, |device| B::int_from_data(data, device))
    }

    fn int_device(tensor: &IntTensor<Self>) -> Device {
        tensor.device()
    }

    fn int_to_device(tensor: IntTensor<Self>, device: &Device) -> IntTensor<Self> {
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

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_into_float(tensor) => Float)
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        multi_tensor_op!(
            (tensor, int),
            (mask, bool),
            (value, int),
            |tensor, mask, value| B::int_mask_where(tensor, mask, value) => Int
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
        multi_tensor_op!(
            (tensor, int),
            (indices, int),
            (value, int),
            |tensor, indices, value| B::int_scatter_add(dim, tensor, indices, value) => Int
        )
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
        multi_tensor_op!(
            (tensor, int),
            (indices, int),
            (value, int),
            |tensor, indices, value| B::int_select_add(tensor, dim, indices, value) => Int
        )
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_equal(lhs, rhs) => Bool)
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_op!(lhs, float, |lhs| B::int_equal_elem(lhs, rhs) => Bool)
    }

    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_greater(lhs, rhs) => Bool)
    }

    fn int_greater_elem(lhs: IntTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_op!(lhs, float, |lhs| B::int_greater_elem(lhs, rhs) => Bool)
    }

    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_greater_equal(lhs, rhs) => Bool)
    }

    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_op!(lhs, float, |lhs| B::int_greater_equal_elem(lhs, rhs) => Bool)
    }

    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_lower(lhs, rhs) => Bool)
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_op!(lhs, float, |lhs| B::int_lower_elem(lhs, rhs) => Bool)
    }

    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_lower_equal(lhs, rhs) => Bool)
    }

    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_op!(lhs, float, |lhs| B::int_lower_equal_elem(lhs, rhs) => Bool)
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_add(lhs, rhs) => Int)
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, float, |lhs| B::int_add_scalar(lhs, rhs) => Int)
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_sub(lhs, rhs) => Int)
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, float, |lhs| B::int_sub_scalar(lhs, rhs) => Int)
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_mul(lhs, rhs) => Int)
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, float, |lhs| B::int_mul_scalar(lhs, rhs) => Int)
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_div(lhs, rhs) => Int)
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, float, |lhs| B::int_div_scalar(lhs, rhs) => Int)
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::int_remainder(lhs, rhs) => Int)
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, float, |lhs| B::int_remainder_scalar(lhs, rhs) => Int)
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
        device: &Device,
    ) -> IntTensor<Self> {
        creation_op!(Int, device, |device| {
            B::int_random(shape, distribution, device)
        })
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::int_expand(tensor, shape) => Int)
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::bitwise_and(lhs, rhs) => Int)
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, float, |lhs| B::bitwise_and_scalar(lhs, rhs) => Int)
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::bitwise_or(lhs, rhs) => Int)
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, float, |lhs| B::bitwise_or_scalar(lhs, rhs) => Int)
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::bitwise_xor(lhs, rhs) => Int)
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, float, |lhs| B::bitwise_xor_scalar(lhs, rhs) => Int)
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_op!(tensor, int, |tensor| B::bitwise_not(tensor) => Int)
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::bitwise_left_shift(lhs, rhs) => Int)
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, float, |lhs| B::bitwise_left_shift_scalar(lhs, rhs) => Int)
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_op!((lhs, int), (rhs, int), |lhs, rhs| B::bitwise_right_shift(lhs, rhs) => Int)
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_op!(lhs, float, |lhs| B::bitwise_right_shift_scalar(lhs, rhs) => Int)
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
}
