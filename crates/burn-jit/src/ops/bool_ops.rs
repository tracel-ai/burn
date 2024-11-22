use crate::{
    element::{BoolElement, ByteElement},
    kernel::{self, LogicalAndOp, LogicalOrOp},
    FloatElement, IntElement, JitBackend, JitRuntime,
};
use burn_tensor::ops::{BoolTensor, ByteTensor, Device, FloatTensor, IntTensor};
use burn_tensor::{ops::BoolTensorOps, Shape, TensorData};
use std::ops::Range;

use super::{expand, permute};

impl<R, F, I, B, P> BoolTensorOps<Self> for JitBackend<R, F, I, B, P>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
    B: BoolElement,
    P: ByteElement,
{
    fn bool_empty(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        super::empty::<R, B>(shape, device)
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> TensorData {
        super::bool_into_data::<R, B>(tensor).await
    }

    fn bool_from_data(data: TensorData, device: &Device<Self>) -> BoolTensor<Self> {
        let data: TensorData = TensorData::new(data.iter::<B>().collect(), data.shape);
        super::from_data::<R, B>(data, device)
    }

    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        kernel::bool_cast::<R, B, I>(tensor)
    }

    fn bool_into_byte(tensor: BoolTensor<Self>) -> ByteTensor<Self> {
        kernel::bool_cast::<R, B, P>(tensor)
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
        kernel::slice::<R, B>(tensor, ranges)
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        ranges: &[Range<usize>],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        kernel::slice_assign::<R, B>(tensor, ranges, value)
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        kernel::equal::<R, B, B>(lhs, rhs)
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        kernel::equal_elem::<R, B, B>(tensor, B::false_val())
    }

    fn bool_or(tensor: BoolTensor<Self>, other: BoolTensor<Self>) -> BoolTensor<Self> {
        kernel::launch_binop::<R, B, LogicalOrOp>(tensor, other)
    }

    fn bool_and(tensor: BoolTensor<Self>, other: BoolTensor<Self>) -> BoolTensor<Self> {
        kernel::launch_binop::<R, B, LogicalAndOp>(tensor, other)
    }

    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        kernel::bool_cast::<R, B, F>(tensor)
    }

    fn bool_swap_dims(mut tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        tensor.strides.swap(dim1, dim2);
        tensor.shape.dims.swap(dim1, dim2);

        tensor
    }

    fn bool_repeat_dim(tensor: BoolTensor<Self>, dim: usize, times: usize) -> BoolTensor<Self> {
        kernel::repeat_dim::<R, B>(tensor, dim, times)
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        permute(tensor, axes)
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        expand(tensor, shape)
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        kernel::flip::<R, B, B>(tensor, axes)
    }
}
