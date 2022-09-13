use crate::{
    backend::tch::{TchBackend, TchDevice, TchTensor},
    backend::Backend,
    ops::TensorOpsDevice,
    TchElement,
};

impl<E: TchElement, const D: usize> TensorOpsDevice<TchBackend<E>, D> for TchTensor<E, D> {
    fn device(&self) -> <TchBackend<E> as Backend>::Device {
        match self.tensor.device() {
            tch::Device::Cpu => TchDevice::Cpu,
            tch::Device::Cuda(num) => TchDevice::Cuda(num),
        }
    }

    fn to_device(&self, device: <TchBackend<E> as Backend>::Device) -> Self {
        let device = match device {
            TchDevice::Cpu => tch::Device::Cpu,
            TchDevice::Cuda(num) => tch::Device::Cuda(num),
        };
        Self {
            kind: self.kind.clone(),
            tensor: self.tensor.to(device),
            shape: self.shape,
        }
    }
}
