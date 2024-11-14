use crate::{kernel, FloatElement, IntElement, JitBackend, JitRuntime};
use burn_tensor::ops::{BoolTensor, Device, FloatTensor, IntTensor};
use burn_tensor::{ops::BoolTensorOps, Shape, TensorData};
use std::ops::Range;

use super::{expand, permute};

impl<R, F, I> BoolTensorOps<Self> for JitBackend<R, F, I>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
{
    fn bool_empty(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        super::empty::<R, u32>(shape, device)
    }

    fn bool_shape(tensor: &BoolTensor<Self>) -> Shape {
        tensor.shape.clone()
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> TensorData {
        super::bool_into_data(tensor).await
    }

    fn bool_from_data(data: TensorData, device: &Device<Self>) -> BoolTensor<Self> {
        let data: TensorData = TensorData::new(data.iter::<u32>().collect(), data.shape);
        super::from_data::<R, u32>(data, device)
    }

    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        kernel::bool_cast::<R, I>(tensor)
    }

    fn bool_device(tensor: &BoolTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn bool_to_device(tensor: BoolTensor<Self>, device: &Device<Self>) -> BoolTensor<Self> {
        super::to_device(tensor, device)
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        super::reshape::<R, u32>(tensor, shape)
    }

    fn bool_slice(tensor: BoolTensor<Self>, ranges: &[Range<usize>]) -> BoolTensor<Self> {
        kernel::slice::<R, u32>(tensor, ranges)
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        ranges: &[Range<usize>],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        kernel::slice_assign::<R, u32>(tensor, ranges, value)
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        kernel::equal::<R, u32>(lhs, rhs)
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        kernel::equal_elem::<R, u32>(tensor, 0)
    }

    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        kernel::bool_cast::<R, F>(tensor)
    }

    fn bool_swap_dims(mut tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        tensor.strides.swap(dim1, dim2);
        tensor.shape.dims.swap(dim1, dim2);

        tensor
    }

    fn bool_repeat_dim(tensor: BoolTensor<Self>, dim: usize, times: usize) -> BoolTensor<Self> {
        kernel::repeat_dim::<R, u32>(tensor, dim, times)
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        permute::<R, u32>(tensor, axes)
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        expand::<R, u32>(tensor, shape)
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        kernel::flip::<R, u32>(tensor, axes)
    }
}
