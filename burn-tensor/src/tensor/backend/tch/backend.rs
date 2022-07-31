use super::TchTensor;
use crate::tensor::{backend::Backend, Element};
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
    type Tensor<const D: usize> = TchTensor<E, D>;
}
