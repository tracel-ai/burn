use super::{TchBackend, TchDevice, TchKind, TchTensor};
use crate::{backend::Backend, ops::TensorOps, Data, ElementConversion, Shape, TchElement};
use std::ops::{Add, Div, Mul, Sub};

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

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: <TchBackend<E> as Backend>::Device,
    ) -> <TchBackend<E> as Backend>::TensorPrimitive<D> {
        let kind = TchKind::<E>::new();
        let tensor =
            tch::Tensor::empty(&shape.dims.map(|a| a as i64), (kind.kind(), device.into()));

        to_tensor(tensor)
    }

    fn add<const D: usize>(lhs: &TchTensor<E, D>, rhs: &TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = (&lhs.tensor).add(&rhs.tensor);

        to_tensor(tensor)
    }

    fn add_scalar<const D: usize>(lhs: &TchTensor<E, D>, rhs: &E) -> TchTensor<E, D> {
        let other: f64 = (rhs.clone()).to_elem();
        let tensor = (&lhs.tensor).add(other).to_kind(lhs.kind.kind());

        to_tensor(tensor)
    }

    fn sub<const D: usize>(lhs: &TchTensor<E, D>, rhs: &TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = (&lhs.tensor).sub(&rhs.tensor);
        to_tensor(tensor)
    }

    fn sub_scalar<const D: usize>(lhs: &TchTensor<E, D>, rhs: &E) -> TchTensor<E, D> {
        let other: f64 = (rhs.clone()).to_elem();
        let tensor = (&lhs.tensor).sub(other).to_kind(lhs.kind.kind());

        to_tensor(tensor)
    }

    fn mul<const D: usize>(lhs: &TchTensor<E, D>, rhs: &TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = (&lhs.tensor).mul(&rhs.tensor);
        to_tensor(tensor)
    }

    fn mul_scalar<const D: usize>(lhs: &TchTensor<E, D>, rhs: &E) -> TchTensor<E, D> {
        let other: f64 = (rhs.clone()).to_elem();
        let tensor = (&lhs.tensor).mul(other).to_kind(lhs.kind.kind());

        to_tensor(tensor)
    }

    fn div<const D: usize>(lhs: &TchTensor<E, D>, rhs: &TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = (&lhs.tensor).div(&rhs.tensor);
        to_tensor(tensor)
    }

    fn div_scalar<const D: usize>(lhs: &TchTensor<E, D>, rhs: &E) -> TchTensor<E, D> {
        let other: f64 = (rhs.clone()).to_elem();
        let tensor = (&lhs.tensor).div(other).to_kind(lhs.kind.kind());

        to_tensor(tensor)
    }

    fn matmul<const D: usize>(lhs: &TchTensor<E, D>, rhs: &TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = lhs.tensor.matmul(&rhs.tensor);
        to_tensor(tensor)
    }

    fn neg<const D: usize>(tensor: &TchTensor<E, D>) -> TchTensor<E, D> {
        Self::mul_scalar(tensor, &(-1f32).to_elem::<E>())
    }
}

fn to_tensor<const D: usize, E: TchElement>(tensor: tch::Tensor) -> TchTensor<E, D> {
    let shape = Shape::from(tensor.size());

    TchTensor {
        tensor,
        shape,
        kind: TchKind::new(),
    }
}
