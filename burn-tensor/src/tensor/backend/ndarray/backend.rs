use super::NdArrayTensor;
use crate::tensor::{
    ops::{TensorOpsDevice, TensorOpsReshape, TensorOpsUtilities},
    Backend, Data, Element, TensorType,
};
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
pub struct NdArrayBackend<E> {
    _e: E,
}

impl<E: Default> Default for NdArrayBackend<E> {
    fn default() -> Self {
        Self { _e: E::default() }
    }
}

impl<E> Backend for NdArrayBackend<E>
where
    E: Element + ScalarOperand + LinalgScalar + SampleUniform,
    Standard: rand::distributions::Distribution<E>,
{
    type E = E;
    type Device = Device;

    fn name() -> String {
        "Nd Array Backend".to_string()
    }
}

impl<E: Element + ScalarOperand + LinalgScalar + SampleUniform, const D: usize> TensorType<D, Self>
    for NdArrayBackend<E>
where
    Standard: rand::distributions::Distribution<E>,
{
    type T = NdArrayTensor<E, D>;

    fn from_data(data: Data<E, D>, _device: Device) -> Self::T {
        let tensor = NdArrayTensor::from_data(data);
        tensor
    }
}
