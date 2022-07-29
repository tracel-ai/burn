use crate::tensor::{
    backend::ndarray::{Device, NdArrayTensor, NdArrayTensorBackend},
    ops::TensorOpsDevice,
    Element,
};
use rand::distributions::Standard;

impl<E, const D: usize> TensorOpsDevice<E, D, NdArrayTensorBackend<E>> for NdArrayTensor<E, D>
where
    E: Element,
    Standard: rand::distributions::Distribution<E>,
{
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn to_device(&self, device: Device) -> Self {
        match device {
            Device::Cpu => self.clone(),
        }
    }
}
