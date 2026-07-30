use crate::{Distribution, Tensor};
use alloc::vec::Vec;

/// Truncated SVD via power iteration with deflation.
///
/// Decomposes `A` (m x n) into `U` (m x k), `S` (k), `Vt` (k x n) such that
/// `A ≈ U @ diag(S) @ Vt`, where `k = min(m, n)`.
///
/// Uses subspace iteration with fixed power steps per component. All
/// normalization and deflation uses pure tensor operations - no per-step
/// CPU synchronizations. Singular values are collected via slice_assign
/// into a pre-allocated tensor.
///
/// # Arguments
/// * `tensor` - An (m x n) 2D tensor.
/// * `iters` - Power iteration steps per component (≥ 3).
///
/// # Returns
/// `(U, S, Vt)` where U is (m x k), S is (k), Vt is (k x n).
///
/// # Performance note
/// Composed operation - O(k · iters · m · n) per component.
/// A backend-kernel SVD is tracked for future optimization.
///
/// # Example
/// ```rust,ignore
/// use burn_tensor::linalg::svd;
/// let a = Tensor::<2>::from_floats([1.0,2.0,3.0,4.0], &device).reshape([2,2]);
/// let (u, s, vt) = svd(a, 10);
/// ```
pub fn svd(tensor: Tensor<2>, iters: usize) -> (Tensor<2>, Tensor<1>, Tensor<2>) {
    let [m, n] = tensor.dims();
    let k = m.min(n);
    let device = tensor.device();
    let iters = iters.max(3);

    let mut u_cols: Vec<Tensor<2>> = Vec::with_capacity(k);
    let mut vt_rows: Vec<Tensor<2>> = Vec::with_capacity(k);
    let mut s_tensor = Tensor::zeros([k], &device);
    let mut residual = tensor;

    for comp in 0..k {
        let mut v = Tensor::<2>::random(
            [n, 1], Distribution::Normal(0.0, 1.0), &device,
        );

        for _ in 0..iters {
            let av = residual.clone().transpose().matmul(residual.clone().matmul(v.clone()));
            let norm = av.clone().powf_scalar(2.0).sum_dim(0).sqrt().clamp_min(1e-12);
            v = av.div(norm);
        }

        let av = residual.clone().matmul(v.clone());
        let sigma = av.clone().powf_scalar(2.0).sum_dim(0).sqrt();

        let u = av.div(sigma.clone());
        residual = residual - u.clone().matmul(v.clone().transpose()).mul(sigma.clone());

        // Store sigma via slice_assign - no CPU sync
        s_tensor = s_tensor.slice_assign(
            [comp..comp+1],
            sigma.reshape([1]),
        );

        u_cols.push(u);
        vt_rows.push(v.transpose());
    }

    let u = Tensor::cat(u_cols, 1);
    let vt = Tensor::cat(vt_rows, 0);
    (u, s_tensor, vt)
}
