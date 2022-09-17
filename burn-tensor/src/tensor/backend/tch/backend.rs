use super::TchTensor;
use crate::tensor::{backend::Backend, TchElement};
use crate::tensor::{Data, Distribution, Shape};

#[derive(Clone, Copy, Debug)]
pub enum TchDevice {
    Cpu,
    Cuda(usize),
}
impl From<TchDevice> for tch::Device {
    fn from(device: TchDevice) -> Self {
        match device {
            TchDevice::Cpu => tch::Device::Cpu,
            TchDevice::Cuda(num) => tch::Device::Cuda(num),
        }
    }
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

impl<E: TchElement> Backend for TchBackend<E> {
    type Device = TchDevice;
    type Elem = E;
    type FullPrecisionElem = f32;
    type FullPrecisionBackend = TchBackend<f32>;
    type IntegerBackend = TchBackend<i64>;
    type TensorPrimitive<const D: usize> = TchTensor<E, D>;
    type BoolTensorPrimitive<const D: usize> = TchTensor<bool, D>;

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: Self::Device,
    ) -> TchTensor<E, D> {
        let device = match device {
            TchDevice::Cpu => tch::Device::Cpu,
            TchDevice::Cuda(num) => tch::Device::Cuda(num),
        };
        TchTensor::from_data(data, device)
    }

    fn from_data_bool<const D: usize>(
        data: Data<bool, D>,
        device: Self::Device,
    ) -> Self::BoolTensorPrimitive<D> {
        let device = match device {
            TchDevice::Cpu => tch::Device::Cpu,
            TchDevice::Cuda(num) => tch::Device::Cuda(num),
        };
        TchTensor::from_data(data, device)
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<Self::Elem>,
        device: Self::Device,
    ) -> Self::TensorPrimitive<D> {
        match distribution {
            Distribution::Standard => {
                let mut tensor = TchTensor::<Self::Elem, D>::empty(shape, device);
                tensor.tensor = tensor.tensor.normal_(0.0, 1.0);
                tensor
            }
            Distribution::Bernoulli(prob) => {
                let mut tensor = TchTensor::<Self::Elem, D>::empty(shape, device);
                tensor.tensor = tensor.tensor.f_bernoulli_float_(prob).unwrap();
                tensor
            }
            Distribution::Uniform(from, to) => {
                let mut tensor = TchTensor::<Self::Elem, D>::empty(shape, device);
                tensor.tensor = tensor
                    .tensor
                    .uniform_(from.to_f64().unwrap(), to.to_f64().unwrap());
                tensor
            }
        }
    }

    fn zeros<const D: usize>(shape: Shape<D>, device: Self::Device) -> Self::TensorPrimitive<D> {
        let mut tensor = TchTensor::<Self::Elem, D>::empty(shape, device);
        tensor.tensor = tensor.tensor.zero_();
        tensor
    }

    fn ones<const D: usize>(shape: Shape<D>, device: Self::Device) -> Self::TensorPrimitive<D> {
        let mut tensor = TchTensor::<Self::Elem, D>::empty(shape, device);
        tensor.tensor = tensor.tensor.ones_like();
        tensor
    }

    fn ad_enabled() -> bool {
        false
    }

    fn name() -> String {
        "tch".to_string()
    }
}
