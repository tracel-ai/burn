use crate::{
    backend::ndarray::{NdArrayBackend, NdArrayDevice, NdArrayTensor},
    ops::TensorOpsDevice,
    NdArrayElement,
};

impl<E, const D: usize> TensorOpsDevice<NdArrayBackend<E>, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
{
    fn device(&self) -> <NdArrayBackend<E> as crate::backend::Backend>::Device {
        NdArrayDevice::Cpu
    }

    fn to_device(&self, _device: <NdArrayBackend<E> as crate::backend::Backend>::Device) -> Self {
        self.clone()
    }
}
