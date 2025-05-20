use crate::ElementConversion;
use crate::backend::Backend;
use crate::tensor::Tensor;

use super::l2_norm;

/// Default epsilon value to avoid division by zero
pub const DEFAULT_EPSILON: f64 = 1e-8;
/// Computes the cosine similarity between two tensors along a specified dimension.
///
/// Calculates the cosine of the angle between inputs as their dot product divided
/// by the product of their L2 norms.
///
/// # Arguments
///
/// * `x1` - First input tensor
/// * `x2` - Second input tensor
/// * `dim` - Dimension along which to compute the similarity
///   (negative indices allowed: -1 for last dimension)
/// * `eps` - Small value to avoid division by zero (default: 1e-8)
///
/// # Returns
///
/// Tensor containing the cosine similarity between x1 and x2
pub fn cosine_similarity<B: Backend, const D: usize>(
    x1: Tensor<B, D>,
    x2: Tensor<B, D>,
    dim: i32,
    eps: Option<B::FloatElem>,
) -> Tensor<B, D> {
    let eps = eps.unwrap_or_else(|| B::FloatElem::from_elem(DEFAULT_EPSILON));

    // Convert negative dimension to positive
    let dim_idx = if dim < 0 { D as i32 + dim } else { dim } as usize;

    // Compute dot product: sum(x1 * x2) along the specified dimension
    let dot_product = (x1.clone() * x2.clone()).sum_dim(dim_idx);

    // Compute L2 norms: ||x1|| and ||x2||
    let norm_x1 = l2_norm(x1, dim_idx);
    let norm_x2 = l2_norm(x2, dim_idx);

    // Calculate the denominator (product of the norms) with epsilon to avoid division by zero
    let denominator = norm_x1.clamp_min(eps) * norm_x2.clamp_min(eps);

    // Return the cosine similarity (dot product divided by the product of norms)
    dot_product / denominator
}
