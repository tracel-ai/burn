use crate::{
    tensor::{IntTensor, IntTensor},
    ADBackendDecorator,
};

use burn_tensor::{
    backend::Backend,
    ops::{IntTensorOps, IntTensorOps},
    Data, Shape,
};

impl<B: Backend> IntTensorOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn int_from_data<const D: usize>(
        data: Data<B::IntElem, D>,
        device: &B::Device,
    ) -> IntTensor<B, D> {
        B::int_from_data(data, device)
    }

    fn int_shape<const D: usize>(tensor: &IntTensor<B, D>) -> Shape<D> {
        B::int_shape(tensor)
    }

    fn int_to_data<const D: usize>(tensor: &IntTensor<B, D>) -> Data<int, D> {
        B::int_to_data(tensor)
    }

    fn int_into_data<const D: usize>(tensor: IntTensor<B, D>) -> Data<int, D> {
        B::int_into_data(tensor)
    }

    fn int_to_device<const D: usize>(
        tensor: IntTensor<B, D>,
        device: &B::Device,
    ) -> IntTensor<B, D> {
        B::int_to_device(tensor, device)
    }

    fn int_device<const D: usize>(tensor: &IntTensor<B, D>) -> B::Device {
        B::int_device(tensor)
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: IntTensor<B, D1>,
        shape: Shape<D2>,
    ) -> IntTensor<B, D2> {
        B::int_reshape(tensor, shape)
    }

    fn int_index<const D1: usize, const D2: usize>(
        tensor: IntTensor<B, D1>,
        indexes: [std::ops::Range<usize>; D2],
    ) -> IntTensor<B, D1> {
        B::int_index(tensor, indexes)
    }

    fn int_empty<const D: usize>(
        shape: Shape<D>,
        device: &<ADBackendDecorator<B> as Backend>::Device,
    ) -> IntTensor<B, D> {
        B::int_empty(shape, device)
    }

    fn int_index_assign<const D1: usize, const D2: usize>(
        tensor: <ADBackendDecorator<B> as Backend>::IntTensorPrimitive<D1>,
        indexes: [std::ops::Range<usize>; D2],
        value: <ADBackendDecorator<B> as Backend>::IntTensorPrimitive<D1>,
    ) -> <ADBackendDecorator<B> as Backend>::IntTensorPrimitive<D1> {
        B::int_index_assign(tensor, indexes, value)
    }

    fn int_cat<const D: usize>(tensors: Vec<IntTensor<B, D>>, dim: usize) -> IntTensor<B, D> {
        B::int_cat(tensors, dim)
    }

    fn int_equal<const D: usize>(lhs: IntTensor<B, D>, rhs: IntTensor<B, D>) -> IntTensor<B, D> {
        B::int_equal(lhs, rhs)
    }

    fn int_equal_elem<const D: usize>(lhs: IntTensor<B, D>, rhs: int) -> IntTensor<B, D> {
        B::int_equal_elem(lhs, rhs)
    }
}
