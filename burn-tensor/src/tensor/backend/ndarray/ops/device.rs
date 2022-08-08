use crate::{
    backend::ndarray::{NdArrayBackend, NdArrayDevice, NdArrayTensor},
    ops::TensorOpsDevice,
    NdArrayElement,
};
use rand::distributions::Standard;

impl<E, const D: usize> TensorOpsDevice<NdArrayBackend<E>, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
    Standard: rand::distributions::Distribution<E>,
{
    fn device(&self) -> <NdArrayBackend<E> as crate::back::Backend>::Device {
        NdArrayDevice::Cpu
    }

    fn to_device(&self, _device: <NdArrayBackend<E> as crate::back::Backend>::Device) -> Self {
        self.clone()
    }
}
