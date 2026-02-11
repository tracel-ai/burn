use burn_backend::{
    ExecutionError, Scalar, TensorData,
    ops::BoolTensorOps,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_std::{Shape, Slice};

use crate::backends::*;
use crate::{Device, Engine, EngineTensor};
use crate::{binary_bool, create_bool, dispatch_async_bool, multi_tensor_op, unary_bool};

impl BoolTensorOps<Self> for Engine {
    fn bool_empty(shape: Shape, device: &Device) -> BoolTensor<Self> {
        create_bool!(device, |device| B::bool_empty(shape, device))
    }

    fn bool_zeros(shape: Shape, device: &Device) -> BoolTensor<Self> {
        create_bool!(device, |device| B::bool_zeros(shape, device))
    }

    fn bool_ones(shape: Shape, device: &Device) -> BoolTensor<Self> {
        create_bool!(device, |device| B::bool_ones(shape, device))
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        dispatch_async_bool!(bool_into_data, tensor)
    }

    fn bool_from_data(data: TensorData, device: &Device) -> BoolTensor<Self> {
        create_bool!(device, |device| B::bool_from_data(data, device))
    }

    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        unary_bool!(bool_into_int, tensor => Int)
    }

    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        unary_bool!(bool_into_float, tensor => Float)
    }

    fn bool_device(tensor: &BoolTensor<Self>) -> Device {
        tensor.device()
    }

    fn bool_to_device(tensor: BoolTensor<Self>, device: &Device) -> BoolTensor<Self> {
        todo!() // TODO: backend bridge
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        unary_bool!(bool_reshape, tensor, shape)
    }

    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        unary_bool!(bool_slice, tensor, slices)
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        slices: &[Slice],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        multi_tensor_op!(Bool, bool(tensor), bool(value), |tensor, value| {
            B::bool_slice_assign(tensor, slices, value)
        })
    }

    fn bool_mask_where(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        multi_tensor_op!(
            Bool,
            bool(tensor),
            bool(mask),
            bool(value),
            |tensor, mask, value| B::bool_mask_where(tensor, mask, value)
        )
    }

    fn bool_mask_fill(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> BoolTensor<Self> {
        multi_tensor_op!(Bool, bool(tensor), bool(mask), |tensor, mask| {
            B::bool_mask_fill(tensor, mask, value)
        })
    }

    fn bool_gather(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        multi_tensor_op!(Bool, bool(tensor), int(indices), |tensor, indices| {
            B::bool_gather(dim, tensor, indices)
        })
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        multi_tensor_op!(
            Bool,
            bool(tensor),
            int(indices),
            bool(value),
            |tensor, indices, value| B::bool_scatter_or(dim, tensor, indices, value)
        )
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        binary_bool!(bool_equal, lhs, rhs)
    }

    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_bool!(bool_equal_elem, lhs, rhs)
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        unary_bool!(bool_not, tensor)
    }

    fn bool_and(tensor: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        binary_bool!(bool_and, tensor, rhs)
    }

    fn bool_or(tensor: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        binary_bool!(bool_or, tensor, rhs)
    }

    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        unary_bool!(bool_swap_dims, tensor, dim1, dim2)
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        unary_bool!(bool_permute, tensor, axes)
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        unary_bool!(bool_flip, tensor, axes)
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        unary_bool!(bool_expand, tensor, shape)
    }

    fn bool_unfold(
        tensor: BoolTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<Self> {
        unary_bool!(bool_unfold, tensor, dim, size, step)
    }
}
