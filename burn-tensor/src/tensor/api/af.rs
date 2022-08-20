use super::Tensor;
use crate::back::Backend;

pub fn relu<const D: usize, B: Backend>(tensor: &Tensor<B, D>) -> Tensor<B, D> {
    tensor.relu()
}

pub fn softmax<const D: usize, B: Backend>(tensor: &Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    log_softmax(&tensor, dim).exp()
}

pub fn log_softmax<const D: usize, B: Backend>(tensor: &Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let tensor_tmp = tensor.exp().sum_dim(dim).log();

    tensor.sub(&tensor_tmp)
}
