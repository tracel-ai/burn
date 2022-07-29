use rand::distributions::Standard;

use crate::tensor::{
    backend::tch::{Device, TchTensor, TchTensorCPUBackend},
    ops::TensorOpsDevice,
    Element,
};

impl<E, const D: usize> TensorOpsDevice<E, D, TchTensorCPUBackend<E>> for TchTensor<E, D>
where
    E: Element,
    Standard: rand::distributions::Distribution<E>,
{
    fn device(&self) -> Device {
        match self.tensor.device() {
            tch::Device::Cpu => Device::Cpu,
            tch::Device::Cuda(num) => Device::Cuda(num),
        }
    }

    fn to_device(&self, device: Device) -> Self {
        let tensor = match device {
            Device::Cpu => self.tensor.to_device(tch::Device::Cpu),
            Device::Cuda(num) => self.tensor.to_device(tch::Device::Cuda(num)),
        };

        let kind = self.kind.clone();
        let shape = self.shape.clone();

        Self {
            tensor,
            kind,
            shape,
        }
    }
}
