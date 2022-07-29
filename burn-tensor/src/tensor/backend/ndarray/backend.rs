use super::NdArrayTensor;
use crate::tensor::{Backend, Data, Element, TensorType};
use ndarray::{LinalgScalar, ScalarOperand};
use rand::distributions::{uniform::SampleUniform, Standard};

#[derive(Debug)]
pub struct NdArrayTensorBackend<E> {
    _e: E,
}

impl<E: Element + ScalarOperand + LinalgScalar + SampleUniform> Backend for NdArrayTensorBackend<E>
where
    Standard: rand::distributions::Distribution<E>,
{
    type E = E;

    fn from_data<const D: usize>(data: Data<E, D>) -> <Self as TensorType<D, Self>>::T
    where
        Self: TensorType<D, Self>,
    {
        <Self as TensorType<D, Self>>::from_data(data)
    }
}

impl<E: Element + ScalarOperand + LinalgScalar + SampleUniform, const D: usize> TensorType<D, Self>
    for NdArrayTensorBackend<E>
where
    Standard: rand::distributions::Distribution<E>,
{
    type T = NdArrayTensor<E, D>;

    fn from_data(data: Data<E, D>) -> Self::T {
        let tensor = NdArrayTensor::from_data(data);
        tensor
    }
}
