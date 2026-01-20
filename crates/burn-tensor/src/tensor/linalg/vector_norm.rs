use crate::backend::Backend;
use crate::tensor::{BasicOps, Tensor};
use crate::{ElementConversion, Numeric};
#[allow(unused_imports)]
use num_traits::float::Float;
/// Specifies the type of norm to compute.
#[derive(Debug, Clone, Copy, PartialEq)]
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

impl Norm {
    /// Get the exponent of the norm.
    pub fn to_exponent(self) -> f64 {
        use Norm::*;
        match self {
            L0 => 0.0,
            L1 => 1.0,
            L2 => 2.0,
            LInf => f64::INFINITY,
            LNegInf => f64::NEG_INFINITY,
            Lp(p) => p,
        }
    }
}

impl From<u32> for Norm {
    fn from(value: u32) -> Self {
        use Norm::*;
        match value {
            0 => L0,
            1 => L1,
            2 => L2,
            u32::MAX => LInf,
            _ => Lp(value as f64),
        }
    }
}

impl From<i32> for Norm {
    fn from(value: i32) -> Self {
        use Norm::*;
        match value {
            0 => L0,
            1 => L1,
            2 => L2,
            i32::MAX => LInf,
            i32::MIN => LNegInf,
            _ => Lp(value as f64),
        }
    }
}

impl From<f32> for Norm {
    fn from(value: f32) -> Self {
        use Norm::*;
        match value {
            0.0 => L0,
            1.0 => L1,
            2.0 => L2,
            f32::INFINITY => LInf,
            f32::NEG_INFINITY => LNegInf,
            _ => Lp(value as f64),
        }
    }
}

impl From<f64> for Norm {
    fn from(value: f64) -> Self {
        use Norm::*;
        match value {
            0.0 => L0,
            1.0 => L1,
            2.0 => L2,
            f64::INFINITY => LInf,
            f64::NEG_INFINITY => LNegInf,
            _ => Lp(value),
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
    lp_norm(x, norm.into().to_exponent(), dim)
}

/// Computes the general ``L(p)`` norm of a tensor along a specified dimension.
///
/// Uses the specialized implementations for:
/// * 0.0
/// * 1.0
/// * 2.0
/// * 2 * N for integral N,
/// * f64::INFINITY,
/// * f64::NEG_INFINITY,
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
    match p {
        0.0 => l0_norm(x, dim),
        1.0 => l1_norm(x, dim),
        2.0 => l2_norm(x, dim),
        p if is_even_integer(p) => lp_signed_norm(x, p as u32, dim),
        f64::INFINITY => max_abs_norm(x, dim),
        f64::NEG_INFINITY => min_abs_norm(x, dim),
        _ => lp_norm_base(x, p, dim),
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
    x.square().sum_dim(dim).sqrt()
}

fn is_even_integer(x: f64) -> bool {
    x.fract() == 0.0 && (x as i64) % 2 == 0
}

/// Computes ``L(2*n)`` for even integer ``n``.
///
/// This lets us skip the abs.
fn lp_signed_norm<B: Backend, const D: usize>(x: Tensor<B, D>, p: u32, dim: usize) -> Tensor<B, D> {
    x.powi_scalar(p).sum_dim(dim).powf_scalar(1. / (p as f64))
}

/// Computes the general ``L(p)`` using the generalized method.
///
/// This uses no specialized implementations and cannot handle:
/// * 0.0
/// * f64::INFINITY,
/// * f64::NEG_INFINITY,
fn lp_norm_base<B: Backend, const D: usize>(x: Tensor<B, D>, p: f64, dim: usize) -> Tensor<B, D> {
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
