use burn_std::FloatDType;

use crate::{AsIndex, check::unwrap_dim_index, tensor::Tensor};

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
    let dim = unwrap_dim_index(dim.try_dim_index(D), "Cosine Similarity");
    let eps = eps.unwrap_or_else(|| {
        x1.dtype()
            .finfo()
            .unwrap_or(FloatDType::F32.finfo())
            .min_positive
    });

    let norm_x1 = l2_norm_impl(x1.clone(), &[dim]).clamp_min(eps);
    let norm_x2 = l2_norm_impl(x2.clone(), &[dim]).clamp_min(eps);

    // Normalize separately: multiplying the clamped norms can underflow to zero,
    // even when epsilon is positive, producing NaN for two zero vectors.
    let x1 = x1 / norm_x1;
    let x2 = x2 / norm_x2;
    (x1 * x2).sum_dim(dim)
}
