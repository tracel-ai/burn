use crate::Numeric;
use crate::backend::Backend;
use crate::tensor::{BasicOps, Tensor};

/// Computes the vector norm of a tensor along a specified dimension.
///
/// The vector norm is defined as:
///
/// - `p = f64::INFINITY`: The max absolute value.
/// - `p = f64::NEG_INFINITY`: The min absolute value.
/// - `p = 0.0`: The count of non-zero elements.
/// - Otherwise, ``sum(abs(x)^p)^(1/p)``
///
/// See:
/// - https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
/// - https://numpy.org/devdocs/reference/generated/numpy.linalg.vector_norm.html
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `p` - The exponent of the vector norm.
/// * `dim` - The dimension to compute the norm over.
///
/// # Returns
///
/// The vector norm of the input tensor.
pub fn vector_norm<B: Backend, const D: usize, K>(
    x: Tensor<B, D, K>,
    p: f64,
    dim: usize,
) -> Tensor<B, D, K>
where
    K: BasicOps<B> + Numeric<B>,
{
    match p {
        f64::INFINITY => x.max_abs_dim(dim),
        f64::NEG_INFINITY => x.abs().min_dim(dim),
        0.0 => {
            // Count non-zero elements
            // TODO: Bool should have sum() support.
            x.zeros_like()
                .mask_fill(x.not_equal_elem(0.0), 1.0)
                .sum_dim(dim)
        }
        1.0 => x.abs().sum_dim(dim),
        _ => x.abs().powf_scalar(p).sum_dim(dim).powf_scalar(1.0 / p),
    }
}

/// Computes the L1 norm of a tensor along a specified dimension.
///
/// This is a convenience function that wraps `vector_norm` with `p = 1.0`.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dim` - The dimension to compute the norm over.
///
/// # Returns
///
/// The L1 norm of the input tensor.
pub fn l1_norm<B: Backend, const D: usize, K>(x: Tensor<B, D, K>, dim: usize) -> Tensor<B, D, K>
where
    K: BasicOps<B> + Numeric<B>,
{
    vector_norm(x, 1.0, dim)
}

/// Computes the L2 norm of a tensor along a specified dimension.
///
/// This is a convenience function that wraps `vector_norm` with `p = 2.0`.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dim` - The dimension to compute the norm over.
///
/// # Returns
///
/// The L2 norm of the input tensor.
pub fn l2_norm<B: Backend, const D: usize, K>(x: Tensor<B, D, K>, dim: usize) -> Tensor<B, D, K>
where
    K: BasicOps<B> + Numeric<B>,
{
    vector_norm(x, 2.0, dim)
}
