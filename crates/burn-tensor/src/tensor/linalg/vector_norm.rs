use crate::check::unwrap_dim_index;
use crate::tensor::Tensor;
use crate::{
    AsIndex, ElementConversion,
    kind::{Numeric, Ordered},
};
use alloc::vec::Vec;
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

/// Computes the vector norm of a tensor along specified dimensions.
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
/// * `dims` - The dimensions to compute the norm over.
///   Negative dimensions are supported and count from the end.
///
/// # Returns
///
/// The vector norm of the input tensor.
pub fn vector_norm_dims<const D: usize, I: AsIndex>(
    x: Tensor<D>,
    norm: impl Into<Norm>,
    dims: &[I],
) -> Tensor<D> {
    if dims.is_empty() {
        return x;
    }
    if dims.len() == 1 {
        let dim = unwrap_dim_index(dims[0].try_dim_index(D), "Vector Norm");
        return vector_norm_impl(x, norm, &[dim]);
    }
    let dims: Vec<usize> = dims
        .iter()
        .map(|&d| unwrap_dim_index(d.try_dim_index(D), "Vector Norm"))
        .collect();
    vector_norm_impl(x, norm, &dims)
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
///   Negative dimensions are supported and count from the end.
///
/// # Returns
///
/// The vector norm of the input tensor.
pub fn vector_norm<const D: usize>(
    x: Tensor<D>,
    norm: impl Into<Norm>,
    dim: impl AsIndex,
) -> Tensor<D> {
    let dim = unwrap_dim_index(dim.try_dim_index(D), "Vector Norm");
    vector_norm_impl(x, norm, &[dim])
}

fn vector_norm_impl<const D: usize>(
    x: Tensor<D>,
    norm: impl Into<Norm>,
    dims: &[usize],
) -> Tensor<D> {
    lp_norm_impl(x, norm.into().to_exponent(), dims)
}

/// Computes the general ``L(p)`` norm of a tensor along specified dimensions.
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
/// * `dims` - The dimensions to compute the norm over.
///   Negative dimensions are supported and count from the end.
///
/// # Returns
///
/// The ``L(p)`` norm of the input tensor.
pub fn lp_norm_dims<const D: usize, I: AsIndex>(x: Tensor<D>, p: f64, dims: &[I]) -> Tensor<D> {
    if dims.is_empty() {
        return x;
    }
    if dims.len() == 1 {
        let dim = unwrap_dim_index(dims[0].try_dim_index(D), "Lp Norm");
        return lp_norm_impl(x, p, &[dim]);
    }
    let dims: Vec<usize> = dims
        .iter()
        .map(|&d| unwrap_dim_index(d.try_dim_index(D), "Lp Norm"))
        .collect();
    lp_norm_impl(x, p, &dims)
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
///   Negative dimensions are supported and count from the end.
///
/// # Returns
///
/// The ``L(p)`` norm of the input tensor.
pub fn lp_norm<const D: usize>(x: Tensor<D>, p: f64, dim: impl AsIndex) -> Tensor<D> {
    let dim = unwrap_dim_index(dim.try_dim_index(D), "Lp Norm");
    lp_norm_impl(x, p, &[dim])
}

fn lp_norm_impl<const D: usize>(x: Tensor<D>, p: f64, dims: &[usize]) -> Tensor<D> {
    match p {
        0.0 => l0_norm_impl(x, dims),
        1.0 => l1_norm_impl(x, dims),
        2.0 => l2_norm_impl(x, dims),
        p if is_even_integer(p) => lp_signed_norm(x, p as u32, dims),
        f64::INFINITY => max_abs_norm_impl(x, dims),
        f64::NEG_INFINITY => min_abs_norm_impl(x, dims),
        _ => lp_norm_base(x, p, dims),
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
///   Negative dimensions are supported and count from the end.
/// * `eps` - The epsilon for the norm.
///
/// # Returns
///
/// The normalized tensor.
pub fn vector_normalize<const D: usize, E: ElementConversion>(
    x: Tensor<D>,
    norm: impl Into<Norm>,
    dim: impl AsIndex,
    eps: E,
) -> Tensor<D> {
    let dim = unwrap_dim_index(dim.try_dim_index(D), "Vector Normalize");
    let norm_tensor = lp_norm_impl(x.clone(), norm.into().to_exponent(), &[dim]).clamp_min(eps);
    x / norm_tensor
}

/// Computes the L0 norm of a tensor along specified dimensions.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dims` - The dimensions to compute the norm over.
///   Negative dimensions are supported and count from the end.
///
/// # Returns
///
/// The L0 norm of the input tensor.
pub fn l0_norm_dims<const D: usize, K, I: AsIndex>(x: Tensor<D, K>, dims: &[I]) -> Tensor<D, K>
where
    K: Numeric,
{
    if dims.is_empty() {
        return x;
    }
    if dims.len() == 1 {
        let dim = unwrap_dim_index(dims[0].try_dim_index(D), "L0 Norm");
        return l0_norm_impl(x, &[dim]);
    }
    let dims: Vec<usize> = dims
        .iter()
        .map(|&d| unwrap_dim_index(d.try_dim_index(D), "L0 Norm"))
        .collect();
    l0_norm_impl(x, &dims)
}

/// Computes the L0 norm of a tensor along a specified dimension.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dim` - The dimension to compute the norm over.
///   Negative dimensions are supported and count from the end.
///
/// # Returns
///
/// The L0 norm of the input tensor.
pub fn l0_norm<const D: usize, K>(x: Tensor<D, K>, dim: impl AsIndex) -> Tensor<D, K>
where
    K: Numeric,
{
    let dim = unwrap_dim_index(dim.try_dim_index(D), "L0 Norm");
    l0_norm_impl(x, &[dim])
}

fn l0_norm_impl<const D: usize, K>(x: Tensor<D, K>, dims: &[usize]) -> Tensor<D, K>
where
    K: Numeric,
{
    x.zeros_like()
        .mask_fill(x.not_equal_scalar(0), 1)
        .sum_dims(dims)
}

/// Computes the L1 norm of a tensor along specified dimensions.
///
/// This is a convenience function that wraps `vector_norm_dims` with `p = 1.0`.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dims` - The dimensions to compute the norm over.
///   Negative dimensions are supported and count from the end.
///
/// # Returns
///
/// The L1 norm of the input tensor.
pub fn l1_norm_dims<const D: usize, K, I: AsIndex>(x: Tensor<D, K>, dims: &[I]) -> Tensor<D, K>
where
    K: Numeric,
{
    if dims.is_empty() {
        return x;
    }
    if dims.len() == 1 {
        let dim = unwrap_dim_index(dims[0].try_dim_index(D), "L1 Norm");
        return l1_norm_impl(x, &[dim]);
    }
    let dims: Vec<usize> = dims
        .iter()
        .map(|&d| unwrap_dim_index(d.try_dim_index(D), "L1 Norm"))
        .collect();
    l1_norm_impl(x, &dims)
}

/// Computes the L1 norm of a tensor along a specified dimension.
///
/// This is a convenience function that wraps `vector_norm` with `p = 1.0`.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dim` - The dimension to compute the norm over.
///   Negative dimensions are supported and count from the end.
///
/// # Returns
///
/// The L1 norm of the input tensor.
pub fn l1_norm<const D: usize, K>(x: Tensor<D, K>, dim: impl AsIndex) -> Tensor<D, K>
where
    K: Numeric,
{
    let dim = unwrap_dim_index(dim.try_dim_index(D), "L1 Norm");
    l1_norm_impl(x, &[dim])
}

fn l1_norm_impl<const D: usize, K>(x: Tensor<D, K>, dims: &[usize]) -> Tensor<D, K>
where
    K: Numeric,
{
    x.abs().sum_dims(dims)
}

/// Computes the L2 norm of a tensor along specified dimensions.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dims` - The dimensions to compute the norm over.
///   Negative dimensions are supported and count from the end.
///
/// # Returns
///
/// The L2 norm of the input tensor.
pub fn l2_norm_dims<const D: usize, I: AsIndex>(x: Tensor<D>, dims: &[I]) -> Tensor<D> {
    if dims.is_empty() {
        return x;
    }
    if dims.len() == 1 {
        let dim = unwrap_dim_index(dims[0].try_dim_index(D), "L2 Norm");
        return l2_norm_impl(x, &[dim]);
    }
    let dims: Vec<usize> = dims
        .iter()
        .map(|&d| unwrap_dim_index(d.try_dim_index(D), "L2 Norm"))
        .collect();
    l2_norm_impl(x, &dims)
}

/// Computes the L2 norm of a tensor along a specified dimension.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dim` - The dimension to compute the norm over.
///   Negative dimensions are supported and count from the end.
///
/// # Returns
///
/// The L2 norm of the input tensor.
pub fn l2_norm<const D: usize>(x: Tensor<D>, dim: impl AsIndex) -> Tensor<D> {
    let dim = unwrap_dim_index(dim.try_dim_index(D), "L2 Norm");
    l2_norm_impl(x, &[dim])
}

pub(super) fn l2_norm_impl<const D: usize>(x: Tensor<D>, dims: &[usize]) -> Tensor<D> {
    x.square().sum_dims(dims).sqrt()
}

fn is_even_integer(x: f64) -> bool {
    x.fract() == 0.0 && (x as i64) % 2 == 0
}

/// Computes ``L(2*n)`` for even integer ``n``.
///
/// This lets us skip the abs.
fn lp_signed_norm<const D: usize>(x: Tensor<D>, p: u32, dims: &[usize]) -> Tensor<D> {
    x.powi_scalar(p).sum_dims(dims).powf_scalar(1. / (p as f64))
}

/// Computes the general ``L(p)`` using the generalized method.
///
/// This uses no specialized implementations and cannot handle:
/// * 0.0
/// * f64::INFINITY,
/// * f64::NEG_INFINITY,
fn lp_norm_base<const D: usize>(x: Tensor<D>, p: f64, dims: &[usize]) -> Tensor<D> {
    x.abs().powf_scalar(p).sum_dims(dims).powf_scalar(1. / p)
}

/// Computes the L:INFINITY norm of a tensor along specified dimensions.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dims` - The dimensions to compute the norm over.
///   Negative dimensions are supported and count from the end.
///
/// # Returns
///
/// The L:INFINITY norm of the input tensor.
pub fn max_abs_norm_dims<const D: usize, K, I: AsIndex>(x: Tensor<D, K>, dims: &[I]) -> Tensor<D, K>
where
    K: Ordered,
{
    if dims.is_empty() {
        return x;
    }
    if dims.len() == 1 {
        let dim = unwrap_dim_index(dims[0].try_dim_index(D), "Max Abs Norm");
        return max_abs_norm_impl(x, &[dim]);
    }
    let dims: Vec<usize> = dims
        .iter()
        .map(|&d| unwrap_dim_index(d.try_dim_index(D), "Max Abs Norm"))
        .collect();
    max_abs_norm_impl(x, &dims)
}

/// Computes the L:INFINITY norm of a tensor along a specified dimension.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dim` - The dimension to compute the norm over.
///   Negative dimensions are supported and count from the end.
///
/// # Returns
///
/// The L:INFINITY norm of the input tensor.
pub fn max_abs_norm<const D: usize, K>(x: Tensor<D, K>, dim: impl AsIndex) -> Tensor<D, K>
where
    K: Ordered,
{
    let dim = unwrap_dim_index(dim.try_dim_index(D), "Max Abs Norm");
    max_abs_norm_impl(x, &[dim])
}

fn max_abs_norm_impl<const D: usize, K>(x: Tensor<D, K>, dims: &[usize]) -> Tensor<D, K>
where
    K: Ordered,
{
    dims.iter()
        .fold(x.abs(), |tensor, &dim| tensor.max_dim(dim))
}

/// Computes the L:NEG_INFINITY norm of a tensor along specified dimensions.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dims` - The dimensions to compute the norm over.
///   Negative dimensions are supported and count from the end.
///
/// # Returns
///
/// The L:NEG_INFINITY norm of the input tensor.
pub fn min_abs_norm_dims<const D: usize, K, I: AsIndex>(x: Tensor<D, K>, dims: &[I]) -> Tensor<D, K>
where
    K: Ordered,
{
    if dims.is_empty() {
        return x;
    }
    if dims.len() == 1 {
        let dim = unwrap_dim_index(dims[0].try_dim_index(D), "Min Abs Norm");
        return min_abs_norm_impl(x, &[dim]);
    }
    let dims: Vec<usize> = dims
        .iter()
        .map(|&d| unwrap_dim_index(d.try_dim_index(D), "Min Abs Norm"))
        .collect();
    min_abs_norm_impl(x, &dims)
}

/// Computes the L:NEG_INFINITY norm of a tensor along a specified dimension.
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dim` - The dimension to compute the norm over.
///   Negative dimensions are supported and count from the end.
///
/// # Returns
///
/// The L:NEG_INFINITY norm of the input tensor.
pub fn min_abs_norm<const D: usize, K>(x: Tensor<D, K>, dim: impl AsIndex) -> Tensor<D, K>
where
    K: Ordered,
{
    let dim = unwrap_dim_index(dim.try_dim_index(D), "Min Abs Norm");
    min_abs_norm_impl(x, &[dim])
}

fn min_abs_norm_impl<const D: usize, K>(x: Tensor<D, K>, dims: &[usize]) -> Tensor<D, K>
where
    K: Ordered,
{
    dims.iter()
        .fold(x.abs(), |tensor, &dim| tensor.min_dim(dim))
}
