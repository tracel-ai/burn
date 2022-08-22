use super::Tensor;
use crate::back::Backend;
use crate::{ElementPrecision, Precision};

pub fn relu<const D: usize, B: Backend>(tensor: &Tensor<B, D>) -> Tensor<B, D> {
    tensor.relu()
}

pub fn softmax<const D: usize, B: Backend>(tensor: &Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    log_softmax(&tensor, dim).exp()
}

pub fn log_softmax<const D: usize, B: Backend>(tensor: &Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let tensor_tmp = match Precision::Half == B::Elem::precision() {
        true => {
            let tensor_full = tensor.to_full_precision();
            let tensor_tmp = tensor_full.exp().sum_dim(dim).log();
            Tensor::from_full_precision(tensor_tmp)
        }
        false => tensor.exp().sum_dim(dim).log(),
    };

    tensor.sub(&tensor_tmp)
}
