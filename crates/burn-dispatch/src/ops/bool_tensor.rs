use burn_backend::{
    ExecutionError, Scalar, TensorData,
    ops::BoolTensorOps,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_std::{Shape, Slice};

use crate::backends::*;
use crate::{Dispatch, DispatchDevice};

impl BoolTensorOps<Self> for Dispatch {
    fn bool_empty(shape: Shape, device: &DispatchDevice) -> BoolTensor<Self> {
        creation_op!(Bool, device, |device| B::bool_empty(shape, device))
    }

    fn bool_zeros(shape: Shape, device: &DispatchDevice) -> BoolTensor<Self> {
        creation_op!(Bool, device, |device| B::bool_zeros(shape, device))
    }

    fn bool_ones(shape: Shape, device: &DispatchDevice) -> BoolTensor<Self> {
        creation_op!(Bool, device, |device| B::bool_ones(shape, device))
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        unary_op!(tensor, bool, |tensor| B::bool_into_data(tensor).await)
    }

    fn bool_from_data(data: TensorData, device: &DispatchDevice) -> BoolTensor<Self> {
        creation_op!(Bool, device, |device| B::bool_from_data(data, device))
    }

    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_into_int(tensor) => Int)
    }

    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_into_float(tensor) => Float)
    }

    fn bool_device(tensor: &BoolTensor<Self>) -> DispatchDevice {
        tensor.device()
    }

    fn bool_to_device(tensor: BoolTensor<Self>, device: &DispatchDevice) -> BoolTensor<Self> {
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
        multi_op!(
            inputs[(tensor, bool), (mask, bool), (value, bool)], => Bool,
            B::bool_mask_where(tensor, mask, value)
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
        multi_op!(
            inputs[(tensor, bool), (indices, int), (value, bool)], => Bool,
            B::bool_scatter_or(dim, tensor, indices, value)
        )
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, bool), (rhs, bool), |lhs, rhs| B::bool_equal(lhs, rhs) => Bool)
    }

    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_op!(lhs, bool, |lhs| B::bool_equal_elem(lhs, rhs) => Bool)
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

    fn bool_select(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        binary_op!((tensor, bool), (indices, int), |tensor, indices| B::bool_select(tensor, dim, indices) => Bool)
    }

    fn bool_select_or(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        multi_op!(
            inputs[(tensor, bool), (indices, int), (value, bool)], => Bool,
            B::bool_select_or(tensor, dim, indices, value)
        )
    }

    fn bool_repeat_dim(tensor: BoolTensor<Self>, dim: usize, times: usize) -> BoolTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_repeat_dim(tensor, dim, times) => Bool)
    }

    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        vec_op!(tensors, bool, |tensors| B::bool_cat(tensors, dim) => Bool)
    }

    fn bool_not_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, bool), (rhs, bool), |lhs, rhs| B::bool_not_equal(lhs, rhs) => Bool)
    }

    fn bool_not_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        unary_op!(lhs, bool, |lhs| B::bool_not_equal_elem(lhs, rhs) => Bool)
    }

    fn bool_xor(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        binary_op!((lhs, bool), (rhs, bool), |lhs, rhs| B::bool_xor(lhs, rhs) => Bool)
    }

    fn bool_transpose(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_transpose(tensor) => Bool)
    }

    fn bool_any(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_any(tensor) => Bool)
    }

    fn bool_any_dim(tensor: BoolTensor<Self>, dim: usize) -> BoolTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_any_dim(tensor, dim) => Bool)
    }

    fn bool_all(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_all(tensor) => Bool)
    }

    fn bool_all_dim(tensor: BoolTensor<Self>, dim: usize) -> BoolTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_all_dim(tensor, dim) => Bool)
    }

    async fn bool_argwhere(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        unary_op!(tensor, bool, |tensor| B::bool_argwhere(tensor).await => Int)
    }
}
