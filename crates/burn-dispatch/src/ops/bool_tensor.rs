use burn_backend::{
    ExecutionError, Scalar, TensorData,
    ops::BoolTensorOps,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_std::{Shape, Slice};

use crate::backends::*;
use crate::{Device, Dispatch};

impl BoolTensorOps<Self> for Dispatch {
    fn bool_empty(shape: Shape, device: &Device) -> BoolTensor<Self> {
        creation_op!(Bool, device, |device| B::bool_empty(shape, device))
    }

    fn bool_zeros(shape: Shape, device: &Device) -> BoolTensor<Self> {
        creation_op!(Bool, device, |device| B::bool_zeros(shape, device))
    }

    fn bool_ones(shape: Shape, device: &Device) -> BoolTensor<Self> {
        creation_op!(Bool, device, |device| B::bool_ones(shape, device))
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        unary_op!(tensor, float, |tensor| B::bool_into_data(tensor).await)
    }

    fn bool_from_data(data: TensorData, device: &Device) -> BoolTensor<Self> {
        creation_op!(Bool, device, |device| B::bool_from_data(data, device))
    }

    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_into_int(tensor) => Int)
    }

    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_into_float(tensor) => Float)
    }

    fn bool_device(tensor: &BoolTensor<Self>) -> Device {
        tensor.device()
    }

    fn bool_to_device(tensor: BoolTensor<Self>, device: &Device) -> BoolTensor<Self> {
        to_device!(
            Bool,
            bool,
            tensor,
            device,
            bool_to_device,
            |inner, device| {
                let data =
                    burn_backend::read_sync(B1::bool_into_data(inner)).expect("Should read data");
                B2::bool_from_data(data, device)
            }
        )
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_reshape(tensor, shape) => Bool)
    }

    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_slice(tensor, slices) => Bool)
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        slices: &[Slice],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        binary_op!((tensor, bool), (value, bool), |tensor, value| B::bool_slice_assign(tensor, slices, value) => Bool)
    }

    fn bool_mask_where(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        multi_tensor_op!(
            (tensor, bool),
            (mask, bool),
            (value, bool),
            |tensor, mask, value| B::bool_mask_where(tensor, mask, value) => Bool
        )
    }

    fn bool_mask_fill(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> BoolTensor<Self> {
        binary_op!((tensor, bool), (mask, bool), |tensor, mask| B::bool_mask_fill(tensor, mask, value) => Bool)
    }

    fn bool_gather(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        binary_op!((tensor, bool), (indices, int), |tensor, indices| B::bool_gather(dim, tensor, indices) => Bool)
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        multi_tensor_op!(
            (tensor, bool),
            (indices, int),
            (value, bool),
            |tensor, indices, value| B::bool_scatter_or(dim, tensor, indices, value) => Bool
        )
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, bool), (rhs, bool), |lhs, rhs| B::bool_equal(lhs, rhs) => Bool)
    }

    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_op!(lhs, float, |lhs| B::bool_equal_elem(lhs, rhs) => Bool)
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_not(tensor) => Bool)
    }

    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, bool), (rhs, bool), |lhs, rhs| B::bool_and(lhs, rhs) => Bool)
    }

    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, bool), (rhs, bool), |lhs, rhs| B::bool_or(lhs, rhs) => Bool)
    }

    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_swap_dims(tensor, dim1, dim2) => Bool)
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_permute(tensor, axes) => Bool)
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_flip(tensor, axes) => Bool)
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_expand(tensor, shape) => Bool)
    }

    fn bool_unfold(
        tensor: BoolTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_unfold(tensor, dim, size, step) => Bool)
    }
}
