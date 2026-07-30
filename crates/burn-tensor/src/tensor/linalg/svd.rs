use crate::Tensor;
use alloc::vec::Vec;

/// Truncated SVD via power iteration with deflation.
///
/// Decomposes `A` (m x n) into `U` (m x k), `S` (k), `Vt` (k x n) such that
/// `A ≈ U @ diag(S) @ Vt`, where `k = min(m, n)`.
///
/// Uses subspace iteration with fixed `iters` power steps per component. The
/// operation is composed from tensor primitives and works on any backend.
///
/// # Arguments
/// * `tensor` — An (m x n) 2D tensor.
/// * `iters` — Power iteration steps per component (default 10).
///
/// # Returns
/// `(U, S, Vt)` where U is (m x k), S is (k), Vt is (k x n).
///
/// # Performance note
/// Each component requires `O(iters · m · n)` composed operations.
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
    let mut s_vals: Vec<f32> = Vec::with_capacity(k);
    let mut vt_rows: Vec<Tensor<2>> = Vec::with_capacity(k);
    let mut residual = tensor;

    for _comp in 0..k {
        let mut v = Tensor::<2>::random(
            [n, 1], crate::Distribution::Normal(0.0, 1.0), &device,
        );

        // Power iteration: v ← (A^T A)^iters · v, normalized each step via tensor ops
        for _ in 0..iters {
            let av = residual.clone().transpose().matmul(residual.clone().matmul(v.clone()));
            let norm = av.clone().powf_scalar(2.0).sum_dim(0).sqrt().clamp_min(1e-12);
            v = av.div(norm);
        }

        let av = residual.clone().matmul(v.clone());
        let sigma = av.clone().powf_scalar(2.0).sum_dim(0).sqrt();
        let sf = f32::from_le_bytes(sigma.clone().into_data().bytes[..4].try_into().unwrap());

        if sf < 1e-10 { break; }

        let u = av.div(sigma.clone());
        s_vals.push(sf);
        residual = residual - u.clone().matmul(v.clone().transpose()).mul(sigma);
        u_cols.push(u);
        vt_rows.push(v.transpose());
    }

    let kf = u_cols.len();
    let u = Tensor::cat(u_cols, 1);
    let vt = Tensor::cat(vt_rows, 0);
    let s = Tensor::<1>::from_floats(s_vals.as_slice(), &device);

    if kf < k {
        (Tensor::cat(vec![u, Tensor::zeros([m, k - kf], &device)], 1),
         Tensor::cat(vec![s, Tensor::zeros([k - kf], &device)], 0),
         Tensor::cat(vec![vt, Tensor::zeros([k - kf, n], &device)], 0))
    } else {
        (u, s, vt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use crate::TestTensorInt as TestTensor;

    #[test]
    fn test_svd_rectangular() {
        let a = TestTensor::<2>::from_data(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], &Default::default(),
        );
        let (u, s, vt) = svd(a, 10);
        assert_eq!(u.dims(), [3, 2]);
        assert_eq!(s.dims(), [2]);
        assert_eq!(vt.dims(), [2, 2]);
    }

    #[test]
    fn test_svd_square() {
        let a = TestTensor::<2>::from_data(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], &Default::default(),
        );
        let (u, s, _vt) = svd(a, 10);
        assert_eq!(u.dims(), [3, 3]);
        assert_eq!(s.dims(), [3]);
    }

    #[test]
    fn test_svd_singular_values_positive() {
        let a = TestTensor::<2>::from_data(
            [[3.0, 0.0], [0.0, 1.0]], &Default::default(),
        );
        let (_u, s, _vt) = svd(a, 20);
        let vals: Vec<f32> = s.into_data().bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        assert!(vals[0] > 2.5, "first SV should be ~3.0, got {}", vals[0]);
    }
}
