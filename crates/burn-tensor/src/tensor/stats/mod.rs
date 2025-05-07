use crate::{Tensor, backend::Backend};

pub fn var<B: Backend, const D: usize>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let mean = tensor.clone().mean_dim(dim);
    var_with_mean(tensor, mean, dim)
}

pub fn var_with_mean<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    mean: Tensor<B, D>,
    dim: usize,
) -> Tensor<B, D> {
    let n = tensor.shape().dims[dim] - 1;
    var_with_mean_n(tensor, mean, dim, n)
}

pub fn var_bias<B: Backend, const D: usize>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let mean = tensor.clone().mean_dim(dim);
    var_with_mean_bias(tensor, mean, dim)
}

pub fn var_with_mean_bias<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    mean: Tensor<B, D>,
    dim: usize,
) -> Tensor<B, D> {
    let n = tensor.shape().dims[dim];
    var_with_mean_n(tensor, mean, dim, n)
}

pub fn var_with_mean_n<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    mean: Tensor<B, D>,
    dim: usize,
    n: usize,
) -> Tensor<B, D> {
    tensor
        .sub(mean)
        .powi_scalar(2)
        .sum_dim(dim)
        .div_scalar(n as f32)
}
