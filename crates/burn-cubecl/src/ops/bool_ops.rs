use crate::{
    element::BoolElement,
    kernel::{self, AndOp, OrOp},
    CubeBackend, CubeRuntime, FloatElement, IntElement,
};
use burn_tensor::ops::{BoolTensor, BoolTensorOps, Device, FloatTensor, IntTensor};
use burn_tensor::{Shape, TensorData};
use std::ops::Range;

use super::{expand, permute};

impl<R, F, I, BT> BoolTensorOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn bool_empty(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        super::empty::<R, BT>(shape, device)
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> TensorData {
        super::into_data::<R, BT>(tensor).await
    }

    fn bool_from_data(data: TensorData, device: &Device<Self>) -> BoolTensor<Self> {
        if data.dtype != BT::dtype() {
            unimplemented!("Unsupported dtype for `bool_from_data`")
        }
        super::from_data::<R>(data, device)
    }

    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        kernel::bool_cast::<R, BT, I>(tensor)
    }

    fn bool_device(tensor: &BoolTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn bool_to_device(tensor: BoolTensor<Self>, device: &Device<Self>) -> BoolTensor<Self> {
        super::to_device(tensor, device)
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        super::reshape(tensor, shape)
    }

    fn bool_slice(tensor: BoolTensor<Self>, ranges: &[Range<usize>]) -> BoolTensor<Self> {
        kernel::slice::<R, BT>(tensor, ranges)
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        ranges: &[Range<usize>],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        kernel::slice_assign::<R, BT>(tensor, ranges, value)
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        kernel::equal::<R, BT, BT>(lhs, rhs)
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        kernel::equal_elem::<R, BT, BT>(tensor, BT::false_val())
    }

    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        kernel::launch_binop::<R, BT, AndOp>(lhs, rhs)
    }

    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        kernel::launch_binop::<R, BT, OrOp>(lhs, rhs)
    }

    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        kernel::bool_cast::<R, BT, F>(tensor)
    }

    fn bool_swap_dims(mut tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        tensor.strides.swap(dim1, dim2);
        tensor.shape.dims.swap(dim1, dim2);

        tensor
    }

    fn bool_repeat_dim(tensor: BoolTensor<Self>, dim: usize, times: usize) -> BoolTensor<Self> {
        kernel::repeat_dim::<R, BT>(tensor, dim, times)
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        permute(tensor, axes)
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        expand(tensor, shape)
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        kernel::flip::<R, BT, BT>(tensor, axes)
    }
}
