use super::NdArrayTensor;
use crate::tensor::{Backend, Data, Element, TensorType};
use ndarray::{LinalgScalar, ScalarOperand};
use rand::distributions::{uniform::SampleUniform, Standard};

#[derive(Debug, Copy, Clone)]
pub enum Device {
    Cpu,
}

impl Default for Device {
    fn default() -> Self {
        Self::Cpu
    }
}

#[derive(Debug)]
pub struct NdArrayTensorBackend<E> {
    _e: E,
}

impl<E: Default> Default for NdArrayTensorBackend<E> {
    fn default() -> Self {
        Self { _e: E::default() }
    }
}

impl<E> Backend for NdArrayTensorBackend<E>
where
    E: Element + ScalarOperand + LinalgScalar + SampleUniform,
    Standard: rand::distributions::Distribution<E>,
{
    type E = E;
    type Device = Device;

    fn from_data<const D: usize>(
        data: Data<E, D>,
        device: Device,
    ) -> <Self as TensorType<D, Self>>::T
    where
        Self: TensorType<D, Self>,
    {
        <Self as TensorType<D, Self>>::from_data(data, device)
    }
}

impl<E: Element + ScalarOperand + LinalgScalar + SampleUniform, const D: usize> TensorType<D, Self>
    for NdArrayTensorBackend<E>
where
    Standard: rand::distributions::Distribution<E>,
{
    type T = NdArrayTensor<E, D>;

    fn from_data(data: Data<E, D>, _device: Device) -> Self::T {
        let tensor = NdArrayTensor::from_data(data);
        tensor
    }
}
