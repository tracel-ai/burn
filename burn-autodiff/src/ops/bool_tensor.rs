use crate::{
    tensor::{ADTensor, BoolTensor, IntTensor},
    ADBackendDecorator,
};

use burn_tensor::{backend::Backend, ops::BoolTensorOps, Data, Shape};

impl<B: Backend> BoolTensorOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn bool_from_data<const D: usize>(data: Data<bool, D>, device: &B::Device) -> BoolTensor<B, D> {
        B::bool_from_data(data, device)
    }

    fn bool_shape<const D: usize>(tensor: &BoolTensor<B, D>) -> Shape<D> {
        B::bool_shape(tensor)
    }

    fn bool_to_data<const D: usize>(tensor: &BoolTensor<B, D>) -> Data<bool, D> {
        B::bool_to_data(tensor)
    }

    fn bool_into_data<const D: usize>(tensor: BoolTensor<B, D>) -> Data<bool, D> {
        B::bool_into_data(tensor)
    }

    fn bool_into_int<const D: usize>(tensor: BoolTensor<B, D>) -> IntTensor<B, D> {
        B::bool_into_int(tensor)
    }

    fn bool_to_device<const D: usize>(
        tensor: BoolTensor<B, D>,
        device: &B::Device,
    ) -> BoolTensor<B, D> {
        B::bool_to_device(tensor, device)
    }

    fn bool_device<const D: usize>(tensor: &BoolTensor<B, D>) -> B::Device {
        B::bool_device(tensor)
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: BoolTensor<B, D1>,
        shape: Shape<D2>,
    ) -> BoolTensor<B, D2> {
        B::bool_reshape(tensor, shape)
    }

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: BoolTensor<B, D1>,
        ranges: [std::ops::Range<usize>; D2],
    ) -> BoolTensor<B, D1> {
        B::bool_slice(tensor, ranges)
    }

    fn bool_empty<const D: usize>(
        shape: Shape<D>,
        device: &<ADBackendDecorator<B> as Backend>::Device,
    ) -> BoolTensor<B, D> {
        B::bool_empty(shape, device)
    }

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: BoolTensor<Self, D1>,
    ) -> BoolTensor<Self, D1> {
        B::bool_slice_assign(tensor, ranges, value)
    }

    fn bool_cat<const D: usize>(tensors: Vec<BoolTensor<B, D>>, dim: usize) -> BoolTensor<B, D> {
        B::bool_cat(tensors, dim)
    }

    fn bool_equal<const D: usize>(
        lhs: BoolTensor<B, D>,
        rhs: BoolTensor<B, D>,
    ) -> BoolTensor<B, D> {
        B::bool_equal(lhs, rhs)
    }

    fn bool_equal_elem<const D: usize>(lhs: BoolTensor<B, D>, rhs: bool) -> BoolTensor<B, D> {
        B::bool_equal_elem(lhs, rhs)
    }

    fn bool_into_float<const D: usize>(
        tensor: BoolTensor<B, D>,
    ) -> <ADBackendDecorator<B> as Backend>::TensorPrimitive<D> {
        ADTensor::new(B::bool_into_float(tensor))
    }
}
