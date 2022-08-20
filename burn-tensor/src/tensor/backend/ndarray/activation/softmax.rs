use crate::tensor::ops::*;
use crate::{activation::Softmax, backend::ndarray::NdArrayTensor, NdArrayElement};
use rand::distributions::Standard;

impl<E, const D: usize> Softmax<E, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
    Standard: rand::distributions::Distribution<E>,
{
    fn softmax(&self, dim: usize) -> Self {
        let exp = self.exp();
        exp.div(&exp.sum_dim(dim))
    }
}
