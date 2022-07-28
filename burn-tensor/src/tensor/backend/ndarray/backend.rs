// use super::NdArrayTensor;
// use crate::tensor::{ops::TensorOpsReshape, Backend, Data, Element, Shape, Tensor, TensorType};
// use ndarray::{LinalgScalar, ScalarOperand};
//
// pub struct NdArrayTensorBackend<E> {
//     _e: E,
// }
//
// impl<E: Element + ScalarOperand + LinalgScalar> Backend for NdArrayTensorBackend<E> {
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
// impl<E: Element + ScalarOperand + LinalgScalar, const D: usize> TensorType<E, D>
//     for NdArrayTensorBackend<E>
// {
//     type T = NdArrayTensor<E, D>;
//
//     fn from_data(data: Data<E, D>) -> Self::T {
//         let tensor = NdArrayTensor::from_data(data);
//         tensor
//     }
// }
//
// fn allo<B: Backend>(tensor: &Tensor<3, B>) {
//     tensor.reshape(Shape::new([1, 3, 4]));
// }
// // impl<E, const D1: usize, const D2: usize> TensorOpsReshape<E, D1, D2, NdArrayTensor<E, D2>>
// //     for Tensor<D1, NdArrayTensorBackend<E>>
// // where
// //     E: Element,
// // {
// //     fn reshape(&self, shape: crate::tensor::Shape<D2>) -> NdArrayTensor<E, D2> {
// //         todo!()
// //     }
// // }
