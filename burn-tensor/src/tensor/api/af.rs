use super::Tensor;
use crate::back::Backend;

pub fn relu<const D: usize, B: Backend>(tensor: &Tensor<B, D>) -> Tensor<B, D> {
    tensor.relu()
}

pub fn softmax<const D: usize, B: Backend>(
    tensor: &Tensor<B, D>,
    dim: usize,
    eps: B::Elem,
) -> Tensor<B, D> {
    let tensor = tensor.add_scalar(&eps);
    let tensor = tensor.exp();
    let tensor = tensor.div(&tensor.sum_dim(dim));

    tensor
}
