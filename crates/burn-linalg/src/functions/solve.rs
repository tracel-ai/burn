use super::lu::{compute_lu_decomposition, swap_tensor_rows};
use crate::{Bool, DType, Tensor};
use alloc::vec;
use burn_std::{FloatDType, Slice};

/// Solves `A @ X = B` for a square, nonsingular matrix `A`.
///
/// `A` has shape `[..., n, n]`. `B` may contain vectors with shape `[..., n]`
/// or matrices with shape `[..., n, k]`. Its batch dimensions may be omitted
/// or broadcast against the batch dimensions of `A`. The output has the
/// broadcast batch dimensions of both inputs.
///
/// Burn uses static tensor ranks, so `DO` is the output rank: `D - 1` for
/// vector right-hand sides and the larger of `D` and `DB` for matrix
/// right-hand sides. For example, `solve::<2, 1, 1>(a, b)` solves one vector,
/// while `solve::<3, 2, 3>(a, b)` broadcasts an unbatched matrix right-hand
/// side across a batch of matrices. If both interpretations fit the input
/// shapes, `DO = D - 1` selects batched vectors (PyTorch convention), and
/// `DO = D` selects a matrix right-hand side (NumPy convention).
///
/// This function uses LU factorization with partial pivoting. It does not
/// compute an inverse or fall back to a least-squares solution. F16 and BF16
/// inputs are computed in F32 and cast back to the input dtype.
///
/// # Panics
///
/// Panics if `A` is not square, if the input shapes cannot be broadcast, if
/// the inputs have different dtypes or devices, or if any matrix in `A` has
/// a zero pivot after LU factorization. `A` and `B` must have real floating
/// point dtypes. A nearly singular matrix with nonzero pivots is accepted,
/// though its solution may be inaccurate.
pub fn solve<const D: usize, const DB: usize, const DO: usize>(
    a: Tensor<D>,
    b: Tensor<DB>,
) -> Tensor<DO> {
    assert!(D >= 2, "linalg::solve: A must have at least two dimensions");
    assert!(DB >= 1, "linalg::solve: B must have at least one dimension");
    let a_dims = a.dims();
    let b_dims = b.dims();
    let n = a_dims[D - 1];
    assert_eq!(a_dims[D - 2], n, "linalg::solve: A must be square");

    let matrix_rhs = DB >= 2 && DO == D.max(DB) && b_dims[DB - 2] == n;
    let vector_rhs = DO == D - 1 && DB < D && b_dims[DB - 1] == n;
    assert!(
        matrix_rhs || vector_rhs,
        "linalg::solve: B must have shape [..., n] or [..., n, k], and DO must match the output rank"
    );
    assert_eq!(a.dtype(), b.dtype(), "linalg::solve: dtypes must match");
    assert_eq!(a.device(), b.device(), "linalg::solve: devices must match");
    assert!(
        !matches!(a.dtype(), DType::QFloat(_)),
        "linalg::solve: inputs must have real floating point dtypes"
    );

    if matrix_rhs {
        solve_impl::<D, DB, DO, DO>(a, b, true)
    } else {
        solve_impl::<D, DB, DO, D>(a, b, false)
    }
}

fn solve_impl<const D: usize, const DB: usize, const DO: usize, const DW: usize>(
    mut a: Tensor<D>,
    mut b: Tensor<DB>,
    matrix_rhs: bool,
) -> Tensor<DO> {
    let a_dims = a.dims();
    let b_dims = b.dims();
    let n = a_dims[D - 1];
    let original_dtype = a.dtype();
    let b_batch_rank = DB - if matrix_rhs { 2 } else { 1 };

    let mut a_reshape = [1; DW];
    a_reshape[DW - D..].copy_from_slice(&a_dims);

    let mut rhs_shape = [1; DW];
    let batch_offset = DW - 2 - b_batch_rank;
    rhs_shape[batch_offset..DW - 2].copy_from_slice(&b_dims[..b_batch_rank]);
    rhs_shape[DW - 2] = n;
    rhs_shape[DW - 1] = if matrix_rhs { b_dims[DB - 1] } else { 1 };

    let mut a_shape = a_reshape;
    let mut b_shape = rhs_shape;
    for dim in 0..DW - 2 {
        let a_size = a_reshape[dim];
        let b_size = rhs_shape[dim];
        assert!(
            a_size == b_size || a_size == 1 || b_size == 1,
            "linalg::solve: batch dimensions are not broadcast-compatible"
        );
        let size = if a_size == 1 { b_size } else { a_size };
        a_shape[dim] = size;
        b_shape[dim] = size;
    }

    let mut output_shape = [1; DO];
    output_shape[..DW - 2].copy_from_slice(&a_shape[..DW - 2]);
    output_shape[DO - 1] = if matrix_rhs { b_shape[DW - 1] } else { n };
    if matrix_rhs {
        output_shape[DO - 2] = n;
    }

    if n == 0 || a_shape[..DW - 2].contains(&0) {
        return Tensor::<DO>::empty(output_shape, (&a.device(), original_dtype));
    }

    let needs_upcast = matches!(original_dtype, DType::F16 | DType::BF16);
    if needs_upcast {
        a = a.cast(FloatDType::F32);
        b = b.cast(FloatDType::F32);
    }
    // Factorize before broadcasting A, so one matrix shared by many right-hand
    // sides is decomposed only once.
    let (lu, pivots) = compute_lu_decomposition(a.reshape(a_reshape));

    let diagonal = Tensor::<DW, Bool>::diag_mask(lu.shape(), 0, &lu.device()).bool_not();
    let singular = lu.clone().equal_scalar(0.0).bool_and(diagonal).any();
    assert!(
        !singular.into_scalar::<bool>(),
        "linalg::solve: A is singular"
    );

    // PyTorch also rejects singular A when B has zero columns.
    if b_shape[DW - 1] == 0 {
        return Tensor::<DO>::empty(output_shape, (&lu.device(), original_dtype));
    }
    let lu = lu.expand(a_shape);
    let mut pivot_shape = a_shape;
    pivot_shape[DW - 1] = 1;
    let pivots = pivots.expand(pivot_shape);
    let mut rhs = b.reshape(rhs_shape).expand(b_shape);

    for row in 0..n {
        let pivot = pivots.clone().slice_dim(DW - 2, row).int();
        rhs = swap_tensor_rows(rhs, pivot, row);
    }

    let mut slices = vec![Slice::full(); DW];
    // Forward substitution for the unit-diagonal lower triangular factor.
    for row in 0..n - 1 {
        let lower = lu
            .clone()
            .slice_dim(DW - 2, row + 1..)
            .slice_dim(DW - 1, row);
        let current = rhs.clone().slice_dim(DW - 2, row);
        let remaining = rhs.clone().slice_dim(DW - 2, row + 1..) - lower.matmul(current);
        slices[DW - 2] = Slice::from(row + 1..);
        rhs = rhs.slice_assign(&slices, remaining);
    }
    slices[DW - 2] = Slice::full();

    // Backward substitution for the upper triangular factor.
    for row in (0..n).rev() {
        let pivot = lu.clone().slice_dim(DW - 2, row).slice_dim(DW - 1, row);
        let current = rhs.clone().slice_dim(DW - 2, row) / pivot;
        slices[DW - 2] = Slice::from(row);
        rhs = rhs.slice_assign(&slices, current.clone());
        if row > 0 {
            let upper = lu.clone().slice_dim(DW - 2, 0..row).slice_dim(DW - 1, row);
            let remaining = rhs.clone().slice_dim(DW - 2, 0..row) - upper.matmul(current);
            slices[DW - 2] = Slice::from(0..row);
            rhs = rhs.slice_assign(&slices, remaining);
        }
    }

    let output = rhs.reshape(output_shape);
    if needs_upcast {
        output.cast(original_dtype)
    } else {
        output
    }
}
