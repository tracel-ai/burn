// use crate::tensor::{backend::TchDevice, Backend, Data, Element, TensorType};
// 
// use super::TchTensor;
// 
// pub struct TchTensorGPUBackend<E, const N: usize> {
//     _e: E,
// }
// 
// impl<E: Element + tch::kind::Element + Into<f64>, const N: usize> Backend
//     for TchTensorGPUBackend<E, N>
// {
//     type E = E;
// 
//     fn from_data<const D: usize>(data: Data<E, D>) -> <Self as TensorType<E, D>>::T
//     where
//         Self: TensorType<E, D>,
//     {
//         <Self as TensorType<E, D>>::from_data(data)
//     }
// }
// 
// impl<E: Element + tch::kind::Element + Into<f64>, const D: usize, const N: usize> TensorType<E, D>
//     for TchTensorGPUBackend<E, N>
// {
//     type T = TchTensor<E, D>;
// 
//     fn from_data(data: Data<E, D>) -> Self::T {
//         let device = TchDevice::Cuda(N);
//         let tensor = TchTensor::from_data(data, device);
//         tensor
//     }
// }
// 
// pub struct TchTensorCPUBackend<E> {
//     _e: E,
// }
// 
// impl<E: Element + tch::kind::Element + Into<f64>> Backend for TchTensorCPUBackend<E> {
//     type E = E;
// 
//     fn from_data<const D: usize>(data: Data<E, D>) -> <Self as TensorType<E, D>>::T
//     where
//         Self: TensorType<E, D>,
//     {
//         <Self as TensorType<E, D>>::from_data(data)
//     }
// }
// 
// impl<E: Element + tch::kind::Element + Into<f64>, const D: usize> TensorType<E, D>
//     for TchTensorCPUBackend<E>
// {
//     type T = TchTensor<E, D>;
// 
//     fn from_data(data: Data<E, D>) -> Self::T {
//         let device = TchDevice::Cpu;
//         let tensor = TchTensor::from_data(data, device);
//         tensor
//     }
// }
