use crate::{
    back::Backend,
    tensor::{
        backend::ndarray::{NdArrayBackend, NdArrayTensor},
        ops::*,
    },
    Data, NdArrayElement,
};
use ndarray::Axis;
use rand::distributions::Standard;

macro_rules! keepdim {
    (
        $D:expr,
        $dim:expr,
        $self:expr,
        mean
    ) => {{
        let tensor: NdArrayTensor<E, $D> = $self.mean_dim($dim);
        let mut shape = $self.shape.clone();
        shape.dims[$dim] = 1;
        tensor.reshape(shape)
    }};
    (
        $D:expr,
        $dim:expr,
        $self:expr,
        sum
    ) => {{
        let tensor: NdArrayTensor<E, $D> = $self.sum_dim($dim);
        let mut shape = $self.shape.clone();
        shape.dims[$dim] = 1;
        tensor.reshape(shape)
    }};
}

impl<E: NdArrayElement, const D: usize> TensorOpsAggregation<NdArrayBackend<E>, D>
    for NdArrayTensor<E, D>
where
    Standard: rand::distributions::Distribution<E>,
{
    fn mean(&self) -> NdArrayTensor<E, 1> {
        let data = Data::from([self.array.mean().unwrap()]);
        NdArrayTensor::from_data(data)
    }

    fn sum(&self) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<1> {
        let data = Data::from([self.array.sum()]);
        NdArrayTensor::from_data(data)
    }

    fn mean_dim<const D2: usize>(&self, dim: usize) -> NdArrayTensor<E, D2> {
        let array = self.array.mean_axis(Axis(dim)).unwrap().into_shared();
        let shape = self.shape.remove_dim(dim);

        NdArrayTensor { array, shape }
    }

    fn sum_dim<const D2: usize>(
        &self,
        dim: usize,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D2> {
        let array = self.array.sum_axis(Axis(dim)).into_shared();
        let shape = self.shape.remove_dim(dim);

        NdArrayTensor { array, shape }
    }

    fn mean_dim_keepdim(&self, dim: usize) -> Self {
        match D {
            2 => keepdim!(1, dim, &self, mean),
            3 => keepdim!(2, dim, &self, mean),
            4 => keepdim!(3, dim, &self, mean),
            5 => keepdim!(4, dim, &self, mean),
            6 => keepdim!(5, dim, &self, mean),
            _ => panic!("Dim not supported {}", D),
        }
    }

    fn sum_dim_keepdim(&self, dim: usize) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        match D {
            2 => keepdim!(1, dim, &self, sum),
            3 => keepdim!(2, dim, &self, sum),
            4 => keepdim!(3, dim, &self, sum),
            5 => keepdim!(4, dim, &self, sum),
            6 => keepdim!(5, dim, &self, sum),
            _ => panic!("Dim not supported {}", D),
        }
    }
}
