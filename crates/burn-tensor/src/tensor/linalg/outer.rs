use crate::backend::Backend;
use crate::indexing::canonicalize_dim;
use crate::tensor::{BasicOps, Tensor};
use crate::{AsIndex, Numeric};

/// Computes the outer product for the last columns of 2 tensors.
///
/// See also: `outer_dim`.
///
/// # Arguments
/// - `lhs`: the "row" tensor, with shape ``[..., M]``.
/// - `rhs`: the "col" tensor, with shape ``[..., N]``.
/// - `dim`: the dimension to product.
///
/// # Returns
/// - the broadcast-shaped ``[..., M, N]`` result.
pub fn outer<B: Backend, const D: usize, const R: usize, K>(
    x: Tensor<B, D, K>,
    y: Tensor<B, D, K>,
) -> Tensor<B, R, K>
where
    K: BasicOps<B> + Numeric<B>,
{
    outer_dim(x, y, -1)
}

/// Computes the dimensional outer product for one dimension of two tensors.
///
/// # Arguments
///
/// - `lhs`: the "row" tensor, with shape ``[..., M, ...]``.
/// - `rhs`: the "col" tensor, with shape ``[..., N, ...]``.
/// - `dim`: the dimension to product, supports negative indexing.
///
/// # Returns
/// - the broadcast-shaped ``[..., M, N, ...]`` result.
///
/// # Panics
/// - if R != D + 1
/// - if the non-product dimensions are not broadcast compatible.
//
// Notes:
// - For large batched inputs, `x_col.matmul(y_row)` *might* be more performant
//   than broadcasted elemwise multiply; benchmarking needed to confirm.
pub fn outer_dim<B: Backend, const D: usize, const R: usize, I: AsIndex, K>(
    lhs: Tensor<B, D, K>,
    rhs: Tensor<B, D, K>,
    dim: I,
) -> Tensor<B, R, K>
where
    K: BasicOps<B> + Numeric<B>,
{
    assert!(
        R == D + 1,
        "`outer` with D={D} must R={} (got R={R})",
        D + 1
    );
    let dim = canonicalize_dim(dim, D, false);

    let x = lhs.unsqueeze_dim::<R>(dim + 1); // (..., m, 1, ...)
    let y = rhs.unsqueeze_dim::<R>(dim); // (..., 1, n, ...)

    x * y // (..., m, n, ...)
}
