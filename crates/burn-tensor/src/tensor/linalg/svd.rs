use crate::{IndexingUpdateOp, Int, Tensor, check, check::TensorCheck};
use alloc::vec;
use alloc::vec::Vec;

/// Computes the singular value decomposition of a square or rectangular matrix using
/// one-sided (Hestenes) Jacobi rotations.
///
/// This function decomposes the input tensor A into three tensors `U`, `S`, `Vt`
/// such that `A = U @ diag(S) @ Vt`, where:
/// - `U` has orthonormal columns of shape `[..., m, k]`
/// - `S` contains the singular values of `A`, sorted in descending order,
///   of shape `[..., k]`
/// - `Vt` has orthonormal rows of shape `[..., k, n]`
///
/// with `k = min(m, n)` (reduced decomposition, matching `torch.linalg.svd`).
///
/// The one-sided (Hestenes) Jacobi algorithm rotates pairs of columns of `A`
/// until the Gram matrix `A^T A` is diagonal. The rotation angle is computed
/// from the column norms `alpha`, `beta` and their dot product `gamma` with
/// the numerically safe formula
/// `t = -sign(zeta) / (|zeta| + sqrt(1 + zeta^2))`, `zeta = (beta - alpha) / (2*gamma)`,
/// which is well defined even for zero or orthogonal columns (no special cases).
/// The right singular vectors are recovered with the LAPACK back-transformation
/// `Vt = diag(1/sigma) U^T A` (two native matmuls).
///
/// Each sweep follows the round-robin tournament schedule, rotating every
/// column pair exactly once; the disjoint pairs of each half-sweep are
/// processed in a single batched pass, so the number of kernel launches is
/// O(n) per sweep rather than O(n^2). The decomposition is exact up to
/// floating-point rounding, with quadratic convergence near the solution.
///
/// All operations are pure tensor operations - no CPU synchronization and no
/// host branching; the number of sweeps is fixed, so the result is
/// deterministic.
///
/// # Arguments
///
/// * `tensor` - The input tensor of shape `[..., m, n]`.
/// * `sweeps` - Number of Jacobi sweeps to run. Each sweep processes every
///   column pair exactly once. 8-15 sweeps reach machine precision for
///   typical f32 matrices; larger values only refine rounding-level effects.
///
/// # Returns
///
/// A tuple of three tensors `(U, S, Vt)`:
/// - `U`: `[..., m, k]` with orthonormal columns
/// - `S`: `[..., k]` singular values in descending order
/// - `Vt`: `[..., k, n]` with orthonormal rows
///
/// # Generic Parameters
///
/// - `D`: The number of dimensions of the input tensor.
/// - `D1`: Must be set to `D - 1` (the rank of the singular value tensor).
///
/// # Panics
///
/// This function will panic if the tensor checks fail:
/// - The input tensor has less than 2 dimensions (`D < 2`).
/// - The input is a quantized tensor with dtype `DType::QFloat`.
/// - The generic parameters do not satisfy `D - 1 == D1`.
///
/// # Example
/// ```rust,ignore
/// use burn_tensor::linalg::svd;
/// use burn_tensor::Tensor;
///
/// fn example() {
///     let device = Default::default();
///     let tensor = Tensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);
///
///     // Compute the singular value decomposition
///     let (u, s, vt) = svd::<2, 1>(tensor, 10);
///
///     // A = U @ diag(S) @ Vt (within tolerance)
///     let recon = u.mul(s.unsqueeze_dim(0)).matmul(vt);
///     println!("{}", recon);
/// }
/// ```
pub fn svd<const D: usize, const D1: usize>(
    tensor: Tensor<D>,
    sweeps: usize,
) -> (Tensor<D>, Tensor<D1>, Tensor<D>) {
    let dims = tensor.dims();
    let original_dtype = tensor.dtype();
    let device = tensor.device();
    check!(TensorCheck::svd_input_tensor::<D, D1>(
        "linalg::svd",
        &dims,
        original_dtype
    ));

    let (n_rows, n_cols) = (dims[D - 2], dims[D - 1]);

    // One-sided Jacobi operates on the column space: require m >= n.
    // For wide matrices decompose A^T and swap the output factors.
    let (mut a, swap) = if n_rows >= n_cols {
        (tensor, false)
    } else {
        (tensor.transpose(), true)
    };
    let (m, n) = (n_rows.max(n_cols), n_rows.min(n_cols));
    let k = n; // min(m, n) after the transpose normalization

    // The right singular vectors are recovered at the end from the original
    // matrix (Vt = diag(1/sigma) U^T A, the LAPACK back-transformation), so
    // the sweeps only rotate the columns of A - no V accumulation.
    let a_orig = a.clone();

    for _ in 0..sweeps {
        // One sweep = a full round-robin tournament: every column pair is
        // rotated exactly once (n-1 half-sweeps for even n, n for odd n).
        let tour_len = if n.is_multiple_of(2) { n - 1 } else { n };
        for t in 0..tour_len {
            let (a_idx, b_idx) = tournament_pairs(n, t);
            let ia = Tensor::from_ints(a_idx.as_slice(), &device);
            let ib = Tensor::from_ints(b_idx.as_slice(), &device);
            a = jacobi_half_sweep(a, ia, ib);
        }
    }

    // Singular values: column norms of the rotated A.
    let sigma = a.clone().powf_scalar(2).sum_dim(D - 2).sqrt(); // [..., 1, n]
    let zeros = sigma.clone().zeros_like();
    let mask = sigma.clone().is_close(zeros, None, None);
    // U = A diag(1/sigma); zero columns map to the zero vector (no NaN).
    let u = a.div(sigma.clone().mask_fill(mask, 1.0)); // [..., m, n]

    // Sort singular values in descending order.
    let sigma_flat = sigma.squeeze_dim::<D1>(D - 2); // [..., k]
    let idx = sigma_flat.clone().argsort_descending(D - 2); // [..., k]

    // Right singular vectors via the LAPACK back-transformation
    // A = U diag(sigma) Vt  ->  Vt = diag(1/sigma) U^T A_orig, computed from
    // the UNSORTED factors; two native matmuls instead of accumulating
    // rotations in the sweeps.
    let inv_sigma = sigma_flat.clone().powf_scalar(-1).mask_fill(
        sigma_flat
            .clone()
            .is_close(sigma_flat.clone().zeros_like(), None, None),
        0.0,
    ); // [..., k]
    let uta = u.clone().transpose().matmul(a_orig); // [..., k, n]
    let vt = uta.mul(inv_sigma.unsqueeze_dim::<D>(D - 1)); // [..., k, n]: scale each row j by 1/sigma_j

    // Sort the factors in descending order of the singular values.
    let mut expand_u = [1; D];
    expand_u[D - 2] = m;
    expand_u[D - 1] = k;
    expand_u[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
    let idx_u = idx
        .clone()
        .unsqueeze_dim::<D>(D - 2)
        .expand::<D, _>(expand_u); // [..., 1, k] -> [..., m, k]
    let u = u.gather(D - 1, idx_u);

    let mut expand_vt = [1; D];
    expand_vt[D - 2] = k;
    expand_vt[D - 1] = n;
    expand_vt[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
    let idx_vt = idx
        .clone()
        .unsqueeze_dim::<D>(D - 1)
        .expand::<D, _>(expand_vt); // [..., k, 1] -> [..., k, n]
    let vt = vt.gather(D - 2, idx_vt);

    let s = sigma_flat.gather(D - 2, idx); // [..., k]

    if swap {
        // The work decomposition is of A^T: A^T = u diag(s) vt_code
        // (with vt_code = vt_work^T). A = vt_code^T diag(s) u^T, so
        // U = vt_code^T and Vt = u^T.
        (vt.transpose(), s, u.transpose())
    } else {
        (u, s, vt)
    }
}

/// One half-sweep of the cyclic Jacobi ordering: rotates the DISJOINT column
/// pairs given by the index tensors `idx0`/`idx1` (the round-robin tournament
/// schedule). All pairs of a half-sweep are independent, so the rotation
/// coefficients are computed in one batched pass and the columns are
/// gathered, rotated, and scattered back in a constant number of kernel
/// launches per half-sweep instead of per pair.
fn jacobi_half_sweep<const D: usize>(
    a: Tensor<D>,
    idx0: Tensor<1, Int>,
    idx1: Tensor<1, Int>,
) -> Tensor<D> {
    let dims = a.dims();
    let m = dims[D - 2];
    let p = idx0.dims()[0];
    if p == 0 {
        return a;
    }

    // Expand the pair indices to [..., m, p].
    let mut reshape_dims = [1; D];
    reshape_dims[D - 1] = p;
    let mut expand_dims = [1; D];
    expand_dims[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
    expand_dims[D - 2] = m;
    expand_dims[D - 1] = p;
    let idx0 = idx0.reshape(reshape_dims).expand(expand_dims);
    let mut expand_dims1 = [1; D];
    expand_dims1[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
    expand_dims1[D - 2] = m;
    expand_dims1[D - 1] = p;
    let idx1 = idx1.reshape(reshape_dims).expand(expand_dims1);

    let x0 = a.clone().gather(D - 1, idx0.clone()); // [..., m, p]
    let x1 = a.clone().gather(D - 1, idx1.clone());

    // Gram entries per pair, batched over p.
    let alpha = x0.clone().powf_scalar(2).sum_dim(D - 2); // [..., 1, p]
    let beta = x1.clone().powf_scalar(2).sum_dim(D - 2);
    let gamma = x0.clone().mul(x1.clone()).sum_dim(D - 2);

    // Numerically safe Jacobi rotation (stable for gamma -> 0 and for
    // zero/orthogonal columns; no special cases required).
    let zeta = beta
        .clone()
        .sub(alpha.clone())
        .div(gamma.clone().mul_scalar(2.0));
    // For the rotation convention [p' q'] = [p q] [[c, -s], [s, c]] the
    // annihilating angle is t = -sign(zeta)/(|zeta| + sqrt(1 + zeta^2)).
    let t = zeta.clone().sign().neg().div(
        zeta.clone()
            .abs()
            .add(zeta.powf_scalar(2).add_scalar(1).sqrt()),
    );
    let c = t.clone().powf_scalar(2).add_scalar(1).powf_scalar(-0.5);
    let s = t.mul(c.clone());
    // gamma == 0 makes zeta = 0/0 (NaN) only when BOTH columns are zero; a
    // no-op rotation is correct there (and equivalent to the t -> 0 limit
    // for already-orthogonal columns).
    let no_rotate = gamma.clone().is_close(gamma.zeros_like(), None, None);
    let c = c.clone().mask_fill(no_rotate.clone(), 1.0);
    let s = s.clone().mask_fill(no_rotate, 0.0);

    // Rotate every pair at once (A <- A J_pq) and write back: subtract the
    // old columns, add the rotated ones (scatter supports only Add).
    let new_x0 = x0.clone().mul(c.clone()).add(x1.clone().mul(s.clone()));
    let new_x1 = x1.clone().mul(c.clone()).sub(x0.clone().mul(s.clone()));
    let a = a.scatter(D - 1, idx0.clone(), x0.neg(), IndexingUpdateOp::Add);
    let a = a.scatter(D - 1, idx1.clone(), x1.neg(), IndexingUpdateOp::Add);
    let a = a.scatter(D - 1, idx0, new_x0, IndexingUpdateOp::Add);
    a.scatter(D - 1, idx1, new_x1, IndexingUpdateOp::Add)
}

/// Round-robin tournament pairing for half-sweep `t` of the cyclic Jacobi
/// ordering: every column pair appears in exactly one half-sweep per
/// tournament. For even `n`, column `n-1` is the anchor paired against the
/// rotating ring `0..n-2`; for odd `n`, column `t` takes the bye.
fn tournament_pairs(n: usize, t: usize) -> (Vec<usize>, Vec<usize>) {
    if n.is_multiple_of(2) {
        let m = n - 1;
        let mut a = vec![t % m];
        let mut b = vec![n - 1];
        for k in 1..(n / 2) {
            a.push((t + k) % m);
            b.push((t + m - k) % m);
        }
        (a, b)
    } else {
        let mut a = Vec::with_capacity((n - 1) / 2);
        let mut b = Vec::with_capacity((n - 1) / 2);
        for k in 1..=(n - 1) / 2 {
            a.push((t + k) % n);
            b.push((t + n - k) % n);
        }
        (a, b)
    }
}
