use super::{TchBackend, TchDevice, TchTensor};
use crate::{backend::Backend, ops::TensorOps, Data, Shape, TchElement};

impl<E: TchElement> TensorOps<TchBackend<E>> for TchBackend<E> {
    fn shape<const D: usize>(tensor: &<TchBackend<E> as Backend>::TensorPrimitive<D>) -> &Shape<D> {
        &tensor.shape
    }

    fn to_data<const D: usize>(
        tensor: &<TchBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> Data<<TchBackend<E> as Backend>::Elem, D> {
        let values: Vec<E> = tensor.tensor.shallow_clone().into();
        Data::new(values, tensor.shape)
    }

    fn into_data<const D: usize>(
        tensor: <TchBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> Data<<TchBackend<E> as Backend>::Elem, D> {
        let values: Vec<E> = tensor.tensor.into();
        Data::new(values, tensor.shape)
    }

    fn bool_shape<const D: usize>(
        tensor: &<TchBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> &Shape<D> {
        &tensor.shape
    }

    fn bool_to_data<const D: usize>(
        tensor: &<TchBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> Data<bool, D> {
        let values: Vec<bool> = tensor.tensor.shallow_clone().into();
        Data::new(values, tensor.shape)
    }

    fn bool_into_data<const D: usize>(
        tensor: <TchBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> Data<bool, D> {
        let values: Vec<bool> = tensor.tensor.into();
        Data::new(values, tensor.shape)
    }
    fn device<const D: usize>(tensor: &TchTensor<E, D>) -> TchDevice {
        match tensor.tensor.device() {
            tch::Device::Cpu => TchDevice::Cpu,
            tch::Device::Cuda(num) => TchDevice::Cuda(num),
        }
    }

    fn to_device<const D: usize>(tensor: &TchTensor<E, D>, device: TchDevice) -> TchTensor<E, D> {
        let device = match device {
            TchDevice::Cpu => tch::Device::Cpu,
            TchDevice::Cuda(num) => tch::Device::Cuda(num),
        };
        TchTensor {
            kind: tensor.kind,
            tensor: tensor.tensor.to(device),
            shape: tensor.shape,
        }
    }
}
