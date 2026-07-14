use crate::tensor::Tensor;
use crate::{AsIndex, kind::Numeric};

/// Computes the outer product for the last columns of 2 tensors.
///
/// See also: [`outer_dim`].
///
/// # Arguments
/// - `lhs`: the "row" tensor, with shape ``[..., i]``.
/// - `rhs`: the "col" tensor, with shape ``[..., j]``.
///
/// # Returns
///
/// A tensor of rank `R = D + 1`, where:
///
/// ``
/// result[..., i, j] = lhs[..., i] * rhs[..., j]
/// ``
pub fn outer<const D: usize, const R: usize, K>(
    lhs: Tensor<D, K>,
    rhs: Tensor<D, K>,
) -> Tensor<R, K>
where
    K: Numeric,
{
    outer_dim(lhs, rhs, -1)
}

/// Computes the outer product along a specific dimension, broadcasting over others.
///
/// For the given `dim`, computes the outer product of elements along that dimension,
/// expanding it into two dimensions of size ``M × N`` at positions ``(dim, dim + 1)``.
///
/// # Arguments
///
/// - `lhs`: left operand, the "row" tensor, with size `M` at dimension `dim`.
/// - `rhs`: right operand, the "col" tensor, with size `N` at dimension `dim`.
/// - `dim`: dimension to compute the outer product along (supports negative indexing).
///
/// # Returns
///
/// A tensor of rank `R = D + 1`, where:
///
/// ``
/// result[..., i, j, ...] = lhs[..., i, ...] * rhs[..., j, ...]
/// ``
//
// Notes:
// - For large batched inputs, `x_col.matmul(y_row)` *might* be more performant
//   than broadcasted elemwise multiply; benchmarking needed to confirm.
pub fn outer_dim<const D: usize, const R: usize, Dim: AsIndex, K>(
    lhs: Tensor<D, K>,
    rhs: Tensor<D, K>,
    dim: Dim,
) -> Tensor<R, K>
where
    K: Numeric,
{
    assert_eq!(
        R,
        D + 1,
        "`outer` with D={D} expects R={} (got R={R})",
        D + 1
    );
    let dim = dim.expect_dim_index(D);

    // (..., i, 1, ...)
    let x = lhs.unsqueeze_dim::<R>(dim + 1);

    // (..., 1, j, ...)
    let y = rhs.unsqueeze_dim::<R>(dim);

    // (..., i, j, ...)
    x * y
}
