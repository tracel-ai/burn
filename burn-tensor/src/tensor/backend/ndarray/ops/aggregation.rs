use crate::{
    backend::Backend,
    tensor::{
        backend::ndarray::{NdArrayBackend, NdArrayTensor},
        ops::*,
    },
    Data, NdArrayElement,
};
use ndarray::Axis;

macro_rules! keepdim {
    (
        $D:expr,
        $dim:expr,
        $self:expr,
        mean
    ) => {{
        let tensor: NdArrayTensor<E, $D> = mean_dim(&$self, $dim);
        let mut shape = $self.shape.clone();
        shape.dims[$dim] = 1;
        NdArrayBackend::reshape(&tensor, shape)
    }};
    (
        $D:expr,
        $dim:expr,
        $self:expr,
        sum
    ) => {{
        let tensor: NdArrayTensor<E, $D> = sum_dim(&$self, $dim);
        let mut shape = $self.shape.clone();
        shape.dims[$dim] = 1;
        NdArrayBackend::reshape(&tensor, shape)
    }};
}

impl<E: NdArrayElement, const D: usize> TensorOpsAggregation<NdArrayBackend<E>, D>
    for NdArrayTensor<E, D>
{
    fn mean(&self) -> NdArrayTensor<E, 1> {
        let data = Data::from([self.array.mean().unwrap()]);
        NdArrayTensor::from_data(data)
    }

    fn sum(&self) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<1> {
        let data = Data::from([self.array.sum()]);
        NdArrayTensor::from_data(data)
    }

    fn mean_dim(&self, dim: usize) -> Self {
        match D {
            1 => keepdim!(0, dim, self, mean),
            2 => keepdim!(1, dim, self, mean),
            3 => keepdim!(2, dim, self, mean),
            4 => keepdim!(3, dim, self, mean),
            5 => keepdim!(4, dim, self, mean),
            6 => keepdim!(5, dim, self, mean),
            _ => panic!("Dim not supported {}", D),
        }
    }

    fn sum_dim(&self, dim: usize) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        match D {
            1 => keepdim!(0, dim, self, sum),
            2 => keepdim!(1, dim, self, sum),
            3 => keepdim!(2, dim, self, sum),
            4 => keepdim!(3, dim, self, sum),
            5 => keepdim!(4, dim, self, sum),
            6 => keepdim!(5, dim, self, sum),
            _ => panic!("Dim not supported {}", D),
        }
    }
}

fn mean_dim<E: NdArrayElement, const D1: usize, const D2: usize>(
    tensor: &NdArrayTensor<E, D1>,
    dim: usize,
) -> NdArrayTensor<E, D2> {
    let array = tensor.array.mean_axis(Axis(dim)).unwrap().into_shared();
    let shape = tensor.shape.remove_dim(dim);

    NdArrayTensor { array, shape }
}

fn sum_dim<E: NdArrayElement, const D1: usize, const D2: usize>(
    tensor: &NdArrayTensor<E, D1>,
    dim: usize,
) -> NdArrayTensor<E, D2> {
    let array = tensor.array.sum_axis(Axis(dim)).into_shared();
    let shape = tensor.shape.remove_dim(dim);

    NdArrayTensor { array, shape }
}
