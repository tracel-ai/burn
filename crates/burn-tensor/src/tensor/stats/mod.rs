use crate::{Int, Tensor};

pub fn var<const D: usize>(tensor: Tensor<D>, dim: usize) -> Tensor<D> {
    let mean = tensor.clone().mean_dim(dim);
    var_with_mean(tensor, mean, dim)
}

pub fn var_with_mean<const D: usize>(tensor: Tensor<D>, mean: Tensor<D>, dim: usize) -> Tensor<D> {
    let n = tensor.shape()[dim].saturating_sub(1);
    var_with_mean_n(tensor, mean, dim, n)
}

pub fn var_bias<const D: usize>(tensor: Tensor<D>, dim: usize) -> Tensor<D> {
    let mean = tensor.clone().mean_dim(dim);
    var_with_mean_bias(tensor, mean, dim)
}

pub fn var_with_mean_bias<const D: usize>(
    tensor: Tensor<D>,
    mean: Tensor<D>,
    dim: usize,
) -> Tensor<D> {
    let n = tensor.shape()[dim];
    var_with_mean_n(tensor, mean, dim, n)
}

pub fn var_with_mean_n<const D: usize>(
    tensor: Tensor<D>,
    mean: Tensor<D>,
    dim: usize,
    n: usize,
) -> Tensor<D> {
    tensor.sub(mean).square().sum_dim(dim).div_scalar(n as f32)
}

pub fn median<const D: usize>(tensor: Tensor<D>, dim: usize) -> Tensor<D> {
    let total_elem_numbers = tensor.dims()[dim];
    // `sort` orders NaN like any other value, so the middle element can be finite even when the
    // slice contains NaN.
    let has_nan = tensor.clone().is_nan().any_dim(dim);
    let sorted_tensor = tensor.sort(dim);

    // Following the PyTorch behavior:
    // - Odd count: the median
    // - Even count: the lower of the two median elements
    //
    // Example:
    // - 5 elements: (5 - 1) / 2 = 4 / 2 = 2
    // - 4 elements: (4 - 1) / 2 = 3 / 2 = 1
    let median_index = (total_elem_numbers - 1) / 2;
    sorted_tensor
        .narrow(dim, median_index, 1)
        .mask_fill(has_nan, f32::NAN)
}

pub fn median_with_indices<const D: usize>(
    tensor: Tensor<D>,
    dim: usize,
) -> (Tensor<D>, Tensor<D, Int>) {
    let total_elem_numbers = tensor.dims()[dim];
    // As in `median`, a slice containing NaN has a NaN median. For such a slice,
    // `max_dim_with_indices` gives NaN and the index of the first NaN.
    let (nan_values, nan_indices) = tensor.clone().max_dim_with_indices(dim);
    let has_nan = nan_values.clone().is_nan();
    let (sorted_tensor, indices) = tensor.sort_with_indices(dim);

    // Following the PyTorch behavior:
    // - Odd count: the median
    // - Even count: the lower of the two median elements
    //
    // Example:
    // - 5 elements: (5 - 1) / 2 = 4 / 2 = 2
    // - 4 elements: (4 - 1) / 2 = 3 / 2 = 1
    let median_index = (total_elem_numbers - 1) / 2;
    let median_values = sorted_tensor
        .narrow(dim, median_index, 1)
        .mask_where(has_nan.clone(), nan_values);
    let median_indices = indices
        .narrow(dim, median_index, 1)
        .mask_where(has_nan, nan_indices);
    (median_values, median_indices)
}
