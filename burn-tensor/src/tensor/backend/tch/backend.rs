use rand::distributions::{uniform::SampleUniform, Standard};

use crate::tensor::{backend::TchDevice, Backend, Data, Element, TensorType};

use super::TchTensor;

// #[derive(Debug, new)]
// pub struct TchTensorGPUBackend<E, const N: usize> {
//     _e: E,
// }
//
// impl<E: Element + tch::kind::Element + Into<f64> + SampleUniform, const N: usize> Backend
//     for TchTensorGPUBackend<E, N>
// where
//     Standard: rand::distributions::Distribution<E>,
// {
//     type E = E;
//
//     fn from_data<const D: usize>(data: Data<E, D>) -> <Self as TensorType<D, Self>>::T
//     where
//         Self: TensorType<D, Self>,
//     {
//         <Self as TensorType<D, Self>>::from_data(data)
//     }
// }
//
// impl<
//         E: Element + tch::kind::Element + Into<f64> + SampleUniform,
//         const D: usize,
//         const N: usize,
//     > TensorType<D, Self> for TchTensorGPUBackend<E, N>
// where
//     Standard: rand::distributions::Distribution<E>,
// {
//     type T = TchTensor<E, D>;
//
//     fn from_data(data: Data<E, D>) -> Self::T {
//         let device = TchDevice::Cuda(N);
//         let tensor = TchTensor::from_data(data, device);
//         tensor
//     }
// }

#[derive(Debug, new)]
pub struct TchTensorCPUBackend<E> {
    _e: E,
}

impl<E: Default> Default for TchTensorCPUBackend<E> {
    fn default() -> Self {
        Self::new(E::default())
    }
}

impl<E: Element + tch::kind::Element + Into<f64> + SampleUniform> Backend for TchTensorCPUBackend<E>
where
    Standard: rand::distributions::Distribution<E>,
{
    type E = E;

    fn from_data<const D: usize>(data: Data<E, D>) -> <Self as TensorType<D, Self>>::T
    where
        Self: TensorType<D, Self>,
    {
        <Self as TensorType<D, Self>>::from_data(data)
    }
}

impl<E: Element + tch::kind::Element + Into<f64> + SampleUniform, const D: usize>
    TensorType<D, Self> for TchTensorCPUBackend<E>
where
    Standard: rand::distributions::Distribution<E>,
{
    type T = TchTensor<E, D>;

    fn from_data(data: Data<E, D>) -> Self::T {
        let device = TchDevice::Cpu;
        let tensor = TchTensor::from_data(data, device);
        tensor
    }
}
