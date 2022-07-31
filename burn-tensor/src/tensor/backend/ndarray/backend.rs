use super::NdArrayTensor;
use crate::tensor::Data;
use crate::tensor::{backend::Backend, Element};
use rand::distributions::Standard;

#[derive(Clone, Copy, Debug)]
pub enum NdArrayDevice {
    Cpu,
}

impl Default for NdArrayDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct NdArrayBackend<E> {
    _e: E,
}

impl<E: Element> Backend for NdArrayBackend<E>
where
    Standard: rand::distributions::Distribution<E>,
{
    type Device = NdArrayDevice;
    type Elem = E;
    type Tensor<const D: usize> = NdArrayTensor<E, D>;

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        _device: Self::Device,
    ) -> NdArrayTensor<E, D> {
        NdArrayTensor::from_data(data)
    }

    fn ad_enabled() -> bool {
        false
    }

    fn name() -> String {
        "ndarray".to_string()
    }
}
