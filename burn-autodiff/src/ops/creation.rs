use crate::tensor::ADTensor;
use burn_tensor::backend::Backend;
use burn_tensor::ops::*;

impl<B: Backend, const D: usize> Zeros for ADTensor<D, B> {
    fn zeros(&self) -> Self {
        ADTensor::from_tensor(self.tensor().zeros())
    }
}

impl<B: Backend, const D: usize> Ones for ADTensor<D, B> {
    fn ones(&self) -> Self {
        ADTensor::from_tensor(self.tensor().ones())
    }
}
