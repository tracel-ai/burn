use burn_std::FloatDType;

use crate::{AsIndex, tensor::Tensor};

use super::vector_norm::l2_norm_impl;

/// Computes the cosine similarity between two tensors along a specified dimension.
///
/// Calculates the cosine of the angle between inputs as their dot product divided
/// by the product of their L2 norms.
///
/// # Arguments
///
/// * `x1` - First input tensor
/// * `x2` - Second input tensor
/// * `dim` - Dimension along which to compute the similarity.
///   Negative dimensions are supported and count from the end.
/// * `eps` - Small value to avoid division by zero (default: dtype's smallest positive normal)
///
/// # Returns
///
/// Tensor containing the cosine similarity between x1 and x2
pub fn cosine_similarity<const D: usize>(
    x1: Tensor<D>,
    x2: Tensor<D>,
    dim: impl AsIndex,
    eps: Option<f64>,
) -> Tensor<D> {
    let dim = dim.expect_dim_index(D);
    let eps = eps.unwrap_or_else(|| {
        x1.dtype()
            .finfo()
            .unwrap_or(FloatDType::F32.finfo())
            .min_positive
    });

    // Compute dot product: sum(x1 * x2) along the specified dimension
    let dot_product = (x1.clone() * x2.clone()).sum_dim(dim);

    // Compute L2 norms: ||x1|| and ||x2||
    let norm_x1 = l2_norm_impl(x1, dim);
    let norm_x2 = l2_norm_impl(x2, dim);

    // Calculate the denominator (product of the norms) with epsilon to avoid division by zero
    let denominator = norm_x1.clamp_min(eps) * norm_x2.clamp_min(eps);

    // Return the cosine similarity (dot product divided by the product of norms)
    dot_product / denominator
}
