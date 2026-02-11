use burn_backend::{
    ExecutionError, Scalar, TensorData,
    ops::IntTensorOps,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_std::{IntDType, Shape, Slice};

use crate::backends::*;
use crate::{Device, Engine, EngineTensor};
use crate::{binary_int, create_int, dispatch_async_int, multi_tensor_op, to_device, unary_int};

impl IntTensorOps<Self> for Engine {
    fn int_empty(shape: Shape, device: &Device, dtype: IntDType) -> IntTensor<Self> {
        create_int!(device, |device| B::int_empty(shape, device, dtype))
    }

    async fn int_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        dispatch_async_int!(int_into_data, tensor)
    }

    fn int_from_data(data: TensorData, device: &Device) -> IntTensor<Self> {
        create_int!(device, |device| B::int_from_data(data, device))
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
        unary_int!(int_reshape, tensor, shape)
    }

    fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self> {
        unary_int!(int_slice, tensor, slices)
    }

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        slices: &[Slice],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        multi_tensor_op!(Int, int(tensor), int(value), |tensor, value| {
            B::int_slice_assign(tensor, slices, value)
        })
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        unary_int!(int_into_float, tensor => Float)
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        multi_tensor_op!(
            Int,
            int(tensor),
            bool(mask),
            int(value),
            |tensor, mask, value| B::int_mask_where(tensor, mask, value)
        )
    }

    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> IntTensor<Self> {
        multi_tensor_op!(Int, int(tensor), bool(mask), |tensor, mask| {
            B::int_mask_fill(tensor, mask, value)
        })
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        multi_tensor_op!(Int, int(tensor), int(indices), |tensor, indices| {
            B::int_gather(dim, tensor, indices)
        })
    }

    fn int_scatter_add(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        multi_tensor_op!(
            Int,
            int(tensor),
            int(indices),
            int(value),
            |tensor, indices, value| B::int_scatter_add(dim, tensor, indices, value)
        )
    }

    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        multi_tensor_op!(Int, int(tensor), int(indices), |tensor, indices| {
            B::int_select(tensor, dim, indices)
        })
    }

    fn int_select_add(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        multi_tensor_op!(
            Int,
            int(tensor),
            int(indices),
            int(value),
            |tensor, indices, value| B::int_select_add(tensor, dim, indices, value)
        )
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_int!(int_equal, lhs, rhs => Bool)
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_int!(int_equal_elem, lhs, rhs => Bool)
    }

    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_int!(int_greater, lhs, rhs => Bool)
    }

    fn int_greater_elem(lhs: IntTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_int!(int_greater_elem, lhs, rhs => Bool)
    }

    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_int!(int_greater_equal, lhs, rhs => Bool)
    }

    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_int!(int_greater_equal_elem, lhs, rhs => Bool)
    }

    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_int!(int_lower, lhs, rhs => Bool)
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_int!(int_lower_elem, lhs, rhs => Bool)
    }

    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        binary_int!(int_lower_equal, lhs, rhs => Bool)
    }

    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_int!(int_lower_equal_elem, lhs, rhs => Bool)
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int!(int_add, lhs, rhs)
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_int!(int_add_scalar, lhs, rhs)
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int!(int_sub, lhs, rhs)
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_int!(int_sub_scalar, lhs, rhs)
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int!(int_mul, lhs, rhs)
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_int!(int_mul_scalar, lhs, rhs)
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int!(int_div, lhs, rhs)
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_int!(int_div_scalar, lhs, rhs)
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int!(int_remainder, lhs, rhs)
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_int!(int_remainder_scalar, lhs, rhs)
    }

    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int!(int_matmul, lhs, rhs)
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int!(int_sum, tensor)
    }

    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_int!(int_sum_dim, tensor, dim)
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int!(int_sum, tensor)
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_int!(int_prod_dim, tensor, dim)
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_int!(int_mean_dim, tensor, dim)
    }

    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_int!(int_cumsum, tensor, dim)
    }

    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_int!(int_cumprod, tensor, dim)
    }

    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_int!(int_cummin, tensor, dim)
    }

    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_int!(int_cummax, tensor, dim)
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_int!(int_argmax, tensor, dim)
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        unary_int!(int_argmin, tensor, dim)
    }

    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int!(int_abs, tensor)
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        unary_int!(int_swap_dims, tensor, dim1, dim2)
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        unary_int!(int_permute, tensor, axes)
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        unary_int!(int_flip, tensor, axes)
    }

    fn int_random(
        shape: Shape,
        distribution: burn_backend::Distribution,
        device: &Device,
    ) -> IntTensor<Self> {
        create_int!(device, |device| B::int_random(shape, distribution, device))
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        unary_int!(int_expand, tensor, shape)
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int!(bitwise_and, lhs, rhs)
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_int!(bitwise_and_scalar, lhs, rhs)
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int!(bitwise_or, lhs, rhs)
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_int!(bitwise_or_scalar, lhs, rhs)
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int!(bitwise_xor, lhs, rhs)
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_int!(bitwise_xor_scalar, lhs, rhs)
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_int!(bitwise_not, tensor)
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int!(bitwise_left_shift, lhs, rhs)
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_int!(bitwise_left_shift_scalar, lhs, rhs)
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        binary_int!(bitwise_right_shift, lhs, rhs)
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: Scalar) -> IntTensor<Self> {
        unary_int!(bitwise_right_shift_scalar, lhs, rhs)
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        unary_int!(int_cast, tensor, dtype)
    }

    fn int_unfold(
        tensor: IntTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Self> {
        unary_int!(int_unfold, tensor, dim, size, step)
    }
}
