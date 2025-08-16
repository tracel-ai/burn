use crate::backend::Backend;
use crate::tensor::{BasicOps, Tensor};
use crate::{ElementConversion, Numeric};

/// Specifies the type of norm to compute.
#[derive(Debug, Clone, Copy)]
pub enum Norm {
    /// L0 norm (count of non-zero elements)
    L0,

    /// L1 norm (sum of absolute values)
    L1,

    /// L2 norm (Euclidean norm)
    L2,

    /// L:INFINITY norm (maximum absolute value)
    LInf,

    /// L:NEG_INFINITY norm (minimum absolute value)
    LNegInf,

    /// Lp norm (generalized norm)
    Lp(f64),
}

impl From<i32> for Norm {
    fn from(value: i32) -> Self {
        match value {
            0 => Norm::L0,
            1 => Norm::L1,
            2 => Norm::L2,
            _ => Norm::Lp(value as f64),
        }
    }
}

impl From<f32> for Norm {
    fn from(value: f32) -> Self {
        match value {
            0.0 => Norm::L0,
            1.0 => Norm::L1,
            2.0 => Norm::L2,
            f32::INFINITY => Norm::LInf,
            f32::NEG_INFINITY => Norm::LNegInf,
            _ => Norm::Lp(value as f64),
        }
    }
}

impl From<f64> for Norm {
    fn from(value: f64) -> Self {
        match value {
            0.0 => Norm::L0,
            1.0 => Norm::L1,
            2.0 => Norm::L2,
            f64::INFINITY => Norm::LInf,
            f64::NEG_INFINITY => Norm::LNegInf,
            _ => Norm::Lp(value),
        }
    }
}

/// Computes the vector norm of a tensor along a specified dimension.
///
/// Generic dispatch wrapper over specialized / optimized norms.
///
/// See:
/// - [torch.linalg.vector_norm](https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html)
/// - [numpy.linalg.vector_norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.vector_norm.html)
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `norm` - The selected norm.
/// * `dim` - The dimension to compute the norm over.
///
/// # Returns
///
/// The vector norm of the input tensor.
pub fn vector_norm<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    norm: impl Into<Norm>,
    dim: usize,
) -> Tensor<B, D> {
    let norm = norm.into();
    match norm {
        Norm::L0 => l0_norm(x, dim),
        Norm::L1 => l1_norm(x, dim),
        Norm::L2 => l2_norm(x, dim),
        Norm::LInf => max_abs_norm(x, dim),
        Norm::LNegInf => min_abs_norm(x, dim),
        Norm::Lp(p) => lp_norm(x, p, dim),
    }
}

/// Normalize a tensor versus its `vector_norm`.
///
/// Equivalent to ``x.clone() / vector_norm(x, norm, dim).clamp_min(eps)``.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `norm` - The selected norm.
/// * `dim` - The dimension to compute the norm over.
/// * `eps` - The epsilon for the norm.
///
/// # Returns
///
/// The normalized tensor.
pub fn vector_normalize<B: Backend, const D: usize, E: ElementConversion>(
    x: Tensor<B, D>,
    norm: impl Into<Norm>,
    dim: usize,
    eps: E,
) -> Tensor<B, D> {
    let norm = vector_norm(x.clone(), norm, dim).clamp_min(eps);
    x / norm
}

/// Computes the L0 norm of a tensor along a specified dimension.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dim` - The dimension to compute the norm over.
///
/// # Returns
///
/// The L0 norm of the input tensor.
pub fn l0_norm<B: Backend, const D: usize, K>(x: Tensor<B, D, K>, dim: usize) -> Tensor<B, D, K>
where
    K: BasicOps<B> + Numeric<B>,
{
    x.zeros_like()
        .mask_fill(x.not_equal_elem(0), 1)
        .sum_dim(dim)
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
    x.abs().sum_dim(dim)
}

/// Computes the L2 norm of a tensor along a specified dimension.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dim` - The dimension to compute the norm over.
///
/// # Returns
///
/// The L2 norm of the input tensor.
pub fn l2_norm<B: Backend, const D: usize>(x: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    x.abs().powi_scalar(2).sum_dim(dim).sqrt()
}

/// Computes the general ``L(p)`` norm of a tensor along a specified dimension.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `p` - The exponent of the Lp norm.
/// * `dim` - The dimension to compute the norm over.
///
/// # Returns
///
/// The ``L(p)`` norm of the input tensor.
pub fn lp_norm<B: Backend, const D: usize>(x: Tensor<B, D>, p: f64, dim: usize) -> Tensor<B, D> {
    x.abs().powf_scalar(p).sum_dim(dim).powf_scalar(1. / p)
}

/// Computes the L:INFINITY norm of a tensor along a specified dimension.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dim` - The dimension to compute the norm over.
///
/// # Returns
///
/// The L:INFINITY norm of the input tensor.
pub fn max_abs_norm<B: Backend, const D: usize, K>(
    x: Tensor<B, D, K>,
    dim: usize,
) -> Tensor<B, D, K>
where
    K: BasicOps<B> + Numeric<B>,
{
    x.max_abs_dim(dim)
}

/// Computes the L:NEG_INFINITY norm of a tensor along a specified dimension.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dim` - The dimension to compute the norm over.
///
/// # Returns
///
/// The L:NEG_INFINITY norm of the input tensor.
pub fn min_abs_norm<B: Backend, const D: usize, K>(
    x: Tensor<B, D, K>,
    dim: usize,
) -> Tensor<B, D, K>
where
    K: BasicOps<B> + Numeric<B>,
{
    x.abs().min_dim(dim)
}
