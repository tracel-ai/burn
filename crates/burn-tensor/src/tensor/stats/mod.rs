use crate::{Tensor, backend::Backend};
use burn_backend::tensor::Int;

pub fn var<B: Backend, const D: usize>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let mean = tensor.clone().mean_dim(dim);
    var_with_mean(tensor, mean, dim)
}

pub fn var_with_mean<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    mean: Tensor<B, D>,
    dim: usize,
) -> Tensor<B, D> {
    let n = tensor.shape()[dim] - 1;
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
    let n = tensor.shape()[dim];
    var_with_mean_n(tensor, mean, dim, n)
}

pub fn var_with_mean_n<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    mean: Tensor<B, D>,
    dim: usize,
    n: usize,
) -> Tensor<B, D> {
    tensor.sub(mean).square().sum_dim(dim).div_scalar(n as f32)
}

pub fn median<B: Backend, const D: usize>(tensor: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let total_elem_numbers = tensor.dims()[dim];
    let sorted_tensor = tensor.sort(dim);

    // Following the PyTorch behavior:
    // - Odd count: the median
    // - Even count: the lower of the two median elements
    //
    // Example:
    // - 5 elements: (5 - 1) / 2 = 4 / 2 = 2
    // - 4 elements: (4 - 1) / 2 = 3 / 2 = 1
    let median_index = (total_elem_numbers - 1) / 2;
    sorted_tensor.narrow(dim, median_index, 1)
}

pub fn median_with_indices<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dim: usize,
) -> (Tensor<B, D>, Tensor<B, D, Int>) {
    let total_elem_numbers = tensor.dims()[dim];
    let (sorted_tensor, indices) = tensor.sort_with_indices(dim);

    // Following the PyTorch behavior:
    // - Odd count: the median
    // - Even count: the lower of the two median elements
    //
    // Example:
    // - 5 elements: (5 - 1) / 2 = 4 / 2 = 2
    // - 4 elements: (4 - 1) / 2 = 3 / 2 = 1
    let median_index = (total_elem_numbers - 1) / 2;
    let median_values = sorted_tensor.narrow(dim, median_index, 1);
    let median_indices = indices.narrow(dim, median_index, 1);
    (median_values, median_indices)
}
