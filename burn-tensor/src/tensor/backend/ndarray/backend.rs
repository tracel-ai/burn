use super::NdArrayTensor;
use crate::tensor::{backend::Backend, NdArrayElement};
use crate::tensor::{Data, Distribution, Shape};

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

impl<E: NdArrayElement> Backend for NdArrayBackend<E> {
    type Device = NdArrayDevice;
    type Elem = E;
    type FullPrecisionElem = f32;
    type FullPrecisionBackend = NdArrayBackend<f32>;
    type TensorPrimitive<const D: usize> = NdArrayTensor<E, D>;
    type BoolTensorPrimitive<const D: usize> = NdArrayTensor<bool, D>;
    type IndexTensorPrimitive<const D: usize> = NdArrayTensor<i64, D>;

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        _device: Self::Device,
    ) -> NdArrayTensor<E, D> {
        NdArrayTensor::from_data(data)
    }

    fn from_data_bool<const D: usize>(
        data: Data<bool, D>,
        _device: Self::Device,
    ) -> Self::BoolTensorPrimitive<D> {
        NdArrayTensor::from_data(data)
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
        "ndarray".to_string()
    }
}
