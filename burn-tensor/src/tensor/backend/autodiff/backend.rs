use rand::distributions::Standard;

use super::ADTensor;
use crate::tensor::backend::Backend;

// #[derive(Clone, Copy, Debug, Default)]
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
