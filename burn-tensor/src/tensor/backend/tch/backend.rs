use super::TchTensor;
use crate::tensor::{backend::Backend, Element};
use crate::tensor::{Data, Distribution, Shape};
use rand::distributions::Standard;

#[derive(Clone, Copy, Debug)]
pub enum TchDevice {
    Cpu,
    Cuda(usize),
}

impl Default for TchDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct TchBackend<E> {
    _e: E,
}

impl<E: Element> Backend for TchBackend<E>
where
    Standard: rand::distributions::Distribution<E>,
{
    type Device = TchDevice;
    type Elem = E;
    type TensorPrimitive<const D: usize> = TchTensor<E, D>;
    type BoolTensorPrimitive<const D: usize> = TchTensor<bool, D>;

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: Self::Device,
    ) -> TchTensor<E, D> {
        let device = match device {
            TchDevice::Cpu => tch::Device::Cpu,
            TchDevice::Cuda(num) => tch::Device::Cuda(num),
        };
        TchTensor::from_data(data, device)
    }

    fn from_data_bool<const D: usize>(
        data: Data<bool, D>,
        device: Self::Device,
    ) -> Self::BoolTensorPrimitive<D> {
        let device = match device {
            TchDevice::Cpu => tch::Device::Cpu,
            TchDevice::Cuda(num) => tch::Device::Cuda(num),
        };
        TchTensor::from_data(data, device)
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<Self::Elem>,
        device: Self::Device,
    ) -> Self::TensorPrimitive<D> {
        Self::from_data(Data::random(shape, distribution), device)
    }

    fn zeros<const D: usize>(shape: Shape<D>, device: Self::Device) -> Self::TensorPrimitive<D> {
        Self::from_data(Data::zeros(shape), device)
    }

    fn ones<const D: usize>(shape: Shape<D>, device: Self::Device) -> Self::TensorPrimitive<D> {
        Self::from_data(Data::ones(shape), device)
    }

    fn ad_enabled() -> bool {
        false
    }

    fn name() -> String {
        "tch".to_string()
    }
}
