use crate::{
    back::Backend,
    tensor::{
        backend::ndarray::{NdArrayBackend, NdArrayTensor},
        ops::*,
        Shape,
    },
    NdArrayElement,
};
use rand::distributions::Standard;

impl<E: NdArrayElement, const D: usize> TensorOpsAggregation<NdArrayBackend<E>, D>
    for NdArrayTensor<E, D>
where
    Standard: rand::distributions::Distribution<E>,
{
    fn mean(&self) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<1> {
        todo!()
    }

    fn sum(&self) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<1> {
        todo!()
    }

    fn mean_dim<const D2: usize>(
        &self,
        dim: usize,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D2> {
        todo!()
    }

    fn sum_dim<const D2: usize>(
        &self,
        dim: usize,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D2> {
        todo!()
    }

    fn mean_dim_keepdim(&self, dim: usize) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        todo!()
    }

    fn sum_dim_keepdim(&self, dim: usize) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        todo!()
    }
}
