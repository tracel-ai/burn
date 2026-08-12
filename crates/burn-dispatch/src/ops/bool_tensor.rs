use alloc::vec::Vec;
use burn_backend::{
    BoolDType, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
    ops::BoolTensorOps,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_backend_extension::backend_dispatch;

use crate::{Dispatch, DispatchDevice};

#[backend_dispatch]
impl BoolTensorOps<Self> for Dispatch {
    #[backend_dispatch(skip)]
    fn bool_empty(shape: Shape, device: &DispatchDevice, dtype: BoolDType) -> BoolTensor<Self> {
        creation_op!(Bool, device, |device| B::bool_empty(shape, device, dtype))
    }

    #[backend_dispatch(skip)]
    fn bool_zeros(shape: Shape, device: &DispatchDevice, dtype: BoolDType) -> BoolTensor<Self> {
        creation_op!(Bool, device, |device| B::bool_zeros(shape, device, dtype))
    }

    #[backend_dispatch(skip)]
    fn bool_ones(shape: Shape, device: &DispatchDevice, dtype: BoolDType) -> BoolTensor<Self> {
        creation_op!(Bool, device, |device| B::bool_ones(shape, device, dtype))
    }

    #[backend_dispatch(skip)]
    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        unary_op!(tensor, bool, |tensor| B::bool_into_data(tensor).await)
    }

    #[backend_dispatch(skip)]
    fn bool_from_data(data: TensorData, device: &DispatchDevice) -> BoolTensor<Self> {
        creation_op!(Bool, device, |device| B::bool_from_data(data, device))
    }

    fn bool_into_int(tensor: BoolTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        B::bool_into_int(tensor, out_dtype)
    }

    fn bool_into_float(tensor: BoolTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self> {
        B::bool_into_float(tensor, out_dtype)
    }

    #[backend_dispatch(skip)]
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
        B::bool_reshape(tensor, shape)
    }

    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        B::bool_slice(tensor, slices)
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        slices: &[Slice],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        B::bool_slice_assign(tensor, slices, value)
    }

    fn bool_mask_where(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        B::bool_mask_where(tensor, mask, value)
    }

    fn bool_mask_fill(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> BoolTensor<Self> {
        B::bool_mask_fill(tensor, mask, value)
    }

    async fn bool_mask_select(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        B::bool_mask_select(tensor, mask).await
    }

    fn bool_gather(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        B::bool_gather(dim, tensor, indices)
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        B::bool_scatter_or(dim, tensor, indices, value)
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        B::bool_equal(lhs, rhs)
    }

    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        B::bool_equal_elem(lhs, rhs)
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        B::bool_not(tensor)
    }

    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        B::bool_and(lhs, rhs)
    }

    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        B::bool_or(lhs, rhs)
    }

    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        B::bool_swap_dims(tensor, dim1, dim2)
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        B::bool_permute(tensor, axes)
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        B::bool_flip(tensor, axes)
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        B::bool_expand(tensor, shape)
    }

    fn bool_unfold(
        tensor: BoolTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<Self> {
        B::bool_unfold(tensor, dim, size, step)
    }

    fn bool_select(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        B::bool_select(tensor, dim, indices)
    }

    fn bool_select_or(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        B::bool_select_or(tensor, dim, indices, value)
    }

    fn bool_repeat_dim(tensor: BoolTensor<Self>, dim: usize, times: usize) -> BoolTensor<Self> {
        B::bool_repeat_dim(tensor, dim, times)
    }

    #[backend_dispatch(skip)]
    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        vec_op!(tensors, bool, |tensors| B::bool_cat(tensors, dim) => Bool)
    }

    fn bool_not_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        B::bool_not_equal(lhs, rhs)
    }

    fn bool_not_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        B::bool_not_equal_elem(lhs, rhs)
    }

    fn bool_xor(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        B::bool_xor(lhs, rhs)
    }

    fn bool_transpose(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        B::bool_transpose(tensor)
    }

    fn bool_any(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        B::bool_any(tensor)
    }

    fn bool_any_dim(tensor: BoolTensor<Self>, dim: usize) -> BoolTensor<Self> {
        B::bool_any_dim(tensor, dim)
    }

    fn bool_all(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        B::bool_all(tensor)
    }

    fn bool_all_dim(tensor: BoolTensor<Self>, dim: usize) -> BoolTensor<Self> {
        B::bool_all_dim(tensor, dim)
    }

    async fn bool_argwhere(tensor: BoolTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        B::bool_argwhere(tensor, out_dtype).await
    }
}
