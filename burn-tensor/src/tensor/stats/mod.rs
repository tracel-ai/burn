use crate::{backend::Backend, ElementConversion, Tensor};

pub fn var<B: Backend, const D: usize>(tensor: &Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let mean = tensor.mean_dim(dim);
    var_with_mean(tensor, &mean, dim)
}

pub fn var_with_mean<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    mean: &Tensor<B, D>,
    dim: usize,
) -> Tensor<B, D> {
    let n = tensor.shape().dims[dim] - 1;
    let n = (n as f32).to_elem();

    tensor.sub(&mean).powf(2.0).sum_dim(dim).div_scalar(&n)
}
