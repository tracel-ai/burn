use super::TchTensor;
use crate::tensor::{Backend, Data, Element, TensorType};
use rand::distributions::{uniform::SampleUniform, Standard};

#[derive(Debug, Copy, Clone)]
pub enum Device {
    Cpu,
    Cuda(usize),
}

impl Default for Device {
    fn default() -> Self {
        Self::Cpu
    }
}

#[derive(Debug, new)]
pub struct TchTensorCPUBackend<E> {
    _e: E,
}

impl<E: Default> Default for TchTensorCPUBackend<E> {
    fn default() -> Self {
        Self::new(E::default())
    }
}

impl<E: Element + tch::kind::Element + Into<f64> + SampleUniform> Backend for TchTensorCPUBackend<E>
where
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

impl<E: Element + tch::kind::Element + Into<f64> + SampleUniform, const D: usize>
    TensorType<D, Self> for TchTensorCPUBackend<E>
where
    Standard: rand::distributions::Distribution<E>,
{
    type T = TchTensor<E, D>;

    fn from_data(data: Data<E, D>, device: Device) -> Self::T {
        let device = match device {
            Device::Cpu => tch::Device::Cpu,
            Device::Cuda(num) => tch::Device::Cuda(num),
        };
        let tensor = TchTensor::from_data(data, device);
        tensor
    }
}
