use super::ADBackendDecorator;
use crate::{backend::Backend, ops::TensorOps, Data, Shape};

impl<B: Backend> TensorOps<ADBackendDecorator<B>> for ADBackendDecorator<B> {
    fn shape<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> &Shape<D> {
        B::shape(tensor.tensor_ref())
    }

    fn to_data<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> Data<<ADBackendDecorator<B> as Backend>::Elem, D> {
        B::to_data(tensor.tensor_ref())
    }

    fn into_data<const D: usize>(
        tensor: <ADBackendDecorator<B> as Backend>::TensorPrimitive<D>,
    ) -> Data<<ADBackendDecorator<B> as Backend>::Elem, D> {
        B::into_data(tensor.tensor())
    }

    fn bool_shape<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D>,
    ) -> &Shape<D> {
        B::bool_shape(tensor)
    }

    fn bool_to_data<const D: usize>(
        tensor: &<ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D>,
    ) -> Data<bool, D> {
        B::bool_to_data(tensor)
    }

    fn bool_into_data<const D: usize>(
        tensor: <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D>,
    ) -> Data<bool, D> {
        B::bool_into_data(tensor)
    }
}
