use super::{BoolTensor, Device, FloatTensor, IntTensor};
use crate::{
    element::{FloatElement, IntElement},
    kernel,
    tensor::WgpuTensor,
    GraphicsApi, WgpuBackend,
};
use burn_tensor::{ops::BoolTensorOps, ops::IntTensorOps, Data, Shape};
use std::ops::Range;

impl<G, F, I> BoolTensorOps<WgpuBackend<G, F, I>> for WgpuBackend<G, F, I>
where
    G: GraphicsApi + 'static,
    F: FloatElement,
    I: IntElement,
{
    fn bool_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> BoolTensor<Self, D> {
        super::empty::<G, u32, D>(shape, device)
    }

    fn bool_shape<const D: usize>(tensor: &BoolTensor<Self, D>) -> Shape<D> {
        tensor.shape.clone()
    }

    fn bool_into_data<const D: usize>(tensor: BoolTensor<Self, D>) -> Data<bool, D> {
        let data = super::into_data(tensor);

        Data::new(data.value.into_iter().map(|i| i != 0).collect(), data.shape)
    }

    fn bool_from_data<const D: usize>(
        data: Data<bool, D>,
        device: &Device<Self>,
    ) -> BoolTensor<Self, D> {
        let data: Data<u32, D> = Data::new(
            data.value
                .into_iter()
                .map(|c| match c {
                    true => 1,
                    false => 0,
                })
                .collect(),
            data.shape,
        );
        super::from_data::<G, u32, D>(data, device)
    }

    fn bool_into_int<const D: usize>(tensor: BoolTensor<Self, D>) -> IntTensor<Self, D> {
        if std::mem::size_of::<I>() == std::mem::size_of::<u32>() {
            return WgpuTensor::new(tensor.context, tensor.shape, tensor.buffer);
        }

        let device = Self::bool_device(&tensor);
        let data = Self::bool_into_data(tensor).convert::<I>();

        Self::int_from_data(data, &device)
    }

    fn bool_device<const D: usize>(tensor: &BoolTensor<Self, D>) -> Device<Self> {
        tensor.context.device.clone()
    }

    fn bool_to_device<const D: usize>(
        tensor: BoolTensor<Self, D>,
        device: &Device<Self>,
    ) -> BoolTensor<Self, D> {
        super::to_device::<G, u32, D>(tensor, device)
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> BoolTensor<Self, D2> {
        super::reshape(tensor, shape)
    }

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> BoolTensor<Self, D1> {
        kernel::slice(tensor, ranges)
    }

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: BoolTensor<Self, D1>,
    ) -> BoolTensor<Self, D1> {
        kernel::slice_assign(tensor, ranges, value)
    }

    fn bool_cat<const D: usize>(
        tensors: Vec<BoolTensor<Self, D>>,
        dim: usize,
    ) -> BoolTensor<Self, D> {
        kernel::cat(tensors, dim)
    }

    fn bool_equal<const D: usize>(
        lhs: BoolTensor<Self, D>,
        rhs: BoolTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::equal(lhs, rhs)
    }

    fn bool_equal_elem<const D: usize>(lhs: BoolTensor<Self, D>, rhs: bool) -> BoolTensor<Self, D> {
        kernel::equal_elem(
            lhs,
            match rhs {
                true => 1,
                false => 0,
            },
        )
    }

    fn bool_into_float<const D: usize>(tensor: BoolTensor<Self, D>) -> FloatTensor<Self, D> {
        kernel::cast(tensor)
    }
}
