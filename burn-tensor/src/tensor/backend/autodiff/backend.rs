// use super::ADTensor;
// use crate::tensor::backend::Backend;
// use rand::distributions::Standard;
//
// #[derive(Clone, Debug, Default)]
// pub struct ADBackend<B: Backend> {
//     _b: B,
// }
//
// impl<B: Backend> Backend for ADBackend<B>
// where
//     Standard: rand::distributions::Distribution<B::Elem>,
// {
//     type Device = B::Device;
//     type Elem = B::Elem;
//     type Tensor<const D: usize> = ADTensor<D, B>;
// }

use super::ADTensor;
use crate::tensor::{
    backend::{
        ndarray::{NdArrayBackend, NdArrayDevice},
        Backend,
    },
    Element,
};
use rand::distributions::Standard;

#[derive(Clone, Copy, Debug, Default)]
pub struct ADBackendNdArray<E> {
    _b: NdArrayBackend<E>,
}

impl<E: Element> Backend for ADBackendNdArray<E>
where
    Standard: rand::distributions::Distribution<E>,
{
    type Device = NdArrayDevice;
    type Elem = E;
    type Tensor<const D: usize> = ADTensor<D, NdArrayBackend<E>>;
}
