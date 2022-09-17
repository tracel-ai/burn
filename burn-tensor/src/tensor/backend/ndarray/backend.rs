use super::NdArrayTensor;
use crate::tensor::Data;
use crate::tensor::{backend::Backend, NdArrayElement};

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
    type IntegerBackend = NdArrayBackend<i64>;
    type TensorPrimitive<const D: usize> = NdArrayTensor<E, D>;
    type BoolTensorPrimitive<const D: usize> = NdArrayTensor<bool, D>;

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

    fn ad_enabled() -> bool {
        false
    }

    fn name() -> String {
        "ndarray".to_string()
    }
}
