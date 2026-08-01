use crate::{DType, Tensor, check, check::TensorCheck};
use alloc::vec;
use alloc::vec::Vec;
use burn_std::{FloatDType, TensorData};
use num_traits::float::Float;

/// Computes the singular value decomposition of a square or rectangular matrix.
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
/// # Algorithm
///
/// Two stages, mirroring the LAPACK `gesvd` structure:
///
/// 1. **Golub-Kahan bidiagonalization** with Householder reflections (the
///    same slice pattern used by `qr`): `A = U1 B V1^T` with `B`
///    upper bidiagonal, `n` reflections on shrinking submatrices.
/// 2. **Bidiagonal QR with shifts** (the LAPACK `dbdsqr` algorithm) to
///    diagonalize `B`. Only `2n` scalars are involved, so this stage runs
///    on the host over the tensor data (deterministic, no kernels) and
///    converges in ~2.5 QR steps per singular value. The accumulated Givens
///    rotations rebuild `U` and `V` from the bidiagonalization factors.
///
/// Both stages are exact to machine precision; the number of QR iterations
/// is governed by a convergence criterion (the `sweeps` argument is kept
/// for API compatibility and bounds the iteration count).
///
/// # Arguments
///
/// * `tensor` - The input tensor of shape `[..., m, n]`.
/// * `sweeps` - Upper bound on the number of QR sweeps (ignored in practice:
///   the algorithm converges in ~2.5 steps per singular value).
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
    mut tensor: Tensor<D>,
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

    // Upcast f16/bf16 to f32 (same convention as `det`), cast back at the end.
    let needs_upcast = original_dtype == DType::F16 || original_dtype == DType::BF16;
    if needs_upcast {
        tensor = tensor.cast(FloatDType::F32);
    }

    // One-sided formulation requires m >= n; decompose A^T for wide matrices.
    let (n_rows, n_cols) = (dims[D - 2], dims[D - 1]);
    let (a, swap) = if n_rows >= n_cols {
        (tensor, false)
    } else {
        (tensor.transpose(), true)
    };
    let (m, n) = (n_rows.max(n_cols), n_rows.min(n_cols));

    // Pull the data and run the whole pipeline on the host: bidiagonalization,
    // dbdsqr diagonalization, Givens accumulation and factor assembly. This is
    // deterministic and backend-independent, and avoids the fused-CUDA
    // per-operation dispatch overhead that dominates tensor-op implementations.
    let batch: usize = dims[..(D - 2)].iter().product();
    let mut dims_u = [1; D];
    dims_u[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
    dims_u[D - 2] = m;
    dims_u[D - 1] = n;
    let mut dims_s = [1; D1];
    if D1 >= 2 {
        dims_s[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
    }
    dims_s[D1 - 1] = n;
    let mut dims_vt = [1; D];
    dims_vt[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
    dims_vt[D - 2] = n;
    dims_vt[D - 1] = n;

    let (u, sigma, vt) = if original_dtype == DType::F64 {
        let a_data = a.into_data().to_vec::<f64>().unwrap();
        let (u, s, vt) = svd_host::<f64>(&a_data, m, n, batch, sweeps);
        (
            Tensor::<D>::from_data(TensorData::new(u, dims_u), &device),
            Tensor::<D1>::from_data(TensorData::new(s, dims_s), &device),
            Tensor::<D>::from_data(TensorData::new(vt, dims_vt), &device),
        )
    } else {
        let a_data = a.into_data().to_vec::<f32>().unwrap();
        let (u, s, vt) = svd_host::<f32>(&a_data, m, n, batch, sweeps);
        (
            Tensor::<D>::from_data(TensorData::new(u, dims_u), &device),
            Tensor::<D1>::from_data(TensorData::new(s, dims_s), &device),
            Tensor::<D>::from_data(TensorData::new(vt, dims_vt), &device),
        )
    };

    // Sort the singular values in descending order and permute the factors.
    let sv = sigma; // singular values, unsorted
    let idx = sv.clone().argsort_descending(D - 2); // [..., n]

    let mut expand_u = [1; D];
    expand_u[D - 2] = m;
    expand_u[D - 1] = n;
    expand_u[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
    let idx_u = idx
        .clone()
        .unsqueeze_dim::<D>(D - 2)
        .expand::<D, _>(expand_u);
    let u = u.gather(D - 1, idx_u);

    let mut expand_vt = [1; D];
    expand_vt[D - 2] = n;
    expand_vt[D - 1] = n;
    expand_vt[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
    let idx_vt = idx
        .clone()
        .unsqueeze_dim::<D>(D - 1)
        .expand::<D, _>(expand_vt);
    let vt = vt.gather(D - 2, idx_vt);

    let sv = sv.gather(D - 2, idx);
    // mask must be computed on the sorted values, its positions move with the gather
    let mask = sv
        .clone()
        .lower_equal(sv.clone().max_dim(D - 2).mul_scalar(1e-6));
    let sv = sv.mask_fill(mask, 0.0);

    let result = if swap {
        (vt.transpose(), sv.clone(), u.transpose())
    } else {
        (u, sv, vt)
    };

    let result = if needs_upcast {
        (
            result.0.cast(original_dtype),
            result.1.cast(original_dtype),
            result.2.cast(original_dtype),
        )
    } else {
        result
    };
    // The composed pipeline is a long op chain; the cubecl CUDA runtime can
    // execute dependent kernels out of order under fusion, so flush once.
    // No-op on eager backends such as ndarray.
    let _ = device.sync();
    result
}

/// Host pipeline: Golub-Kahan bidiagonalization + dbdsqr + factor assembly,
/// per batch element. Pure scalar math over tensor data, deterministic and
/// identical on every backend.
fn svd_host<F: Float + Copy>(
    a: &[F],
    m: usize,
    n: usize,
    batch: usize,
    max_sweeps: usize,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    let mut u = vec![F::zero(); batch * m * n];
    let mut sigma = vec![F::zero(); batch * n];
    let mut vt = vec![F::zero(); batch * n * n];
    let mut d = vec![F::zero(); n];
    let mut e = vec![F::zero(); n.saturating_sub(1)];
    let mut givens: Vec<(usize, F, F, F, F)> = Vec::new();

    for b in 0..batch {
        let (mut u1, bv, mut v1) = bidiag_host(&a[b * m * n..(b + 1) * m * n], m, n);
        for i in 0..n {
            d[i] = bv[i * n + i];
        }
        for i in 0..n.saturating_sub(1) {
            e[i] = bv[i * n + i + 1];
        }
        givens.clear();
        let sigma_b = dbdsqr(&mut d, &mut e, &mut givens, max_sweeps);

        // U = U1 @ (product of left Givens rotations): apply each rotation to
        // the column pair of U1 directly (same operator as the accumulation).
        for &(k, cl, sl, _, _) in &givens {
            for i in 0..m {
                let (a0, b0) = (u1[i * m + k], u1[i * m + k + 1]);
                u1[i * m + k] = cl * a0 + sl * b0;
                u1[i * m + k + 1] = -sl * a0 + cl * b0;
            }
        }
        // Vt = (V1 @ (product of right Givens rotations))^T.
        for &(k, _, _, cr, sr) in &givens {
            for i in 0..n {
                let (a0, b0) = (v1[i * n + k], v1[i * n + k + 1]);
                v1[i * n + k] = cr * a0 + sr * b0;
                v1[i * n + k + 1] = -sr * a0 + cr * b0;
            }
        }
        // Absorb the signs of the diagonal into U.
        for k in 0..n {
            if d[k] < F::zero() {
                for i in 0..m {
                    u1[i * m + k] = -u1[i * m + k];
                }
            }
        }
        for i in 0..m {
            for j in 0..n {
                u[b * m * n + i * n + j] = u1[i * m + j];
            }
        }
        for i in 0..n {
            for j in 0..n {
                vt[b * n * n + i * n + j] = v1[j * n + i];
            }
            sigma[b * n + i] = sigma_b[i];
        }
    }
    (u, sigma, vt)
}

/// Golub-Kahan bidiagonalization on the host: `A = U1 B V1^T` with `B` upper
/// bidiagonal (row-major `[m, n]`), using Householder reflections on
/// shrinking submatrices, mirroring the tensor-op version operation for
/// operation.
fn bidiag_host<F: Float + Copy>(a: &[F], m: usize, n: usize) -> (Vec<F>, Vec<F>, Vec<F>) {
    let mut u1 = vec![F::zero(); m * m];
    for i in 0..m {
        u1[i * m + i] = F::one();
    }
    let mut v1 = vec![F::zero(); n * n];
    for i in 0..n {
        v1[i * n + i] = F::one();
    }
    let mut a = a.to_vec();

    for i in 0..n {
        // Left reflection: annihilate the subdiagonal of column i.
        let norm2: F = (i..m).fold(F::zero(), |s, k| s + a[k * n + i] * a[k * n + i]);
        let norm = norm2.sqrt();
        let x0 = a[i * n + i];
        // sign = -(sign(x0)), with zero mapping to -1 (mask_fill in the tensor version).
        let sign = if x0 >= F::zero() { -F::one() } else { F::one() };
        let u0 = x0 - norm * sign;
        let tau = if norm == F::zero() {
            F::zero()
        } else {
            -u0 / (norm * sign)
        };
        if norm != F::zero() {
            let mut w = vec![F::zero(); m];
            w[i] = F::one();
            for k in (i + 1)..m {
                w[k] = a[k * n + i] / u0;
            }
            // wta[j] = w^T a[:, j], then a_new = a - tau w wta.
            for j in i..n {
                let wta: F = (i..m).fold(F::zero(), |s, k| s + w[k] * a[k * n + j]);
                for k in i..m {
                    a[k * n + j] = a[k * n + j] - tau * w[k] * wta;
                }
            }
            // U1 = U1 (I - tau w w^T): update the columns i..m.
            for i2 in 0..m {
                let uw: F = (i..m).fold(F::zero(), |s, k| s + u1[i2 * m + k] * w[k]);
                if uw != F::zero() {
                    for k in i..m {
                        u1[i2 * m + k] = u1[i2 * m + k] - tau * uw * w[k];
                    }
                }
            }
        }

        // Right reflection: annihilate row i right of the superdiagonal.
        if i + 1 < n - 1 {
            let norm2: F = ((i + 1)..n).fold(F::zero(), |s, j| s + a[i * n + j] * a[i * n + j]);
            let norm = norm2.sqrt();
            let y0 = a[i * n + i + 1];
            let sign = if y0 >= F::zero() { -F::one() } else { F::one() };
            let u0 = y0 - norm * sign;
            let tau = if norm == F::zero() {
                F::zero()
            } else {
                -u0 / (norm * sign)
            };
            if norm != F::zero() {
                let mut w = vec![F::zero(); n];
                w[i + 1] = F::one();
                for j in (i + 2)..n {
                    w[j] = a[i * n + j] / u0;
                }
                // aw[k] = a[k, :] w, then a_new = a - tau aw w^T.
                for k in i..m {
                    let aw: F = ((i + 1)..n).fold(F::zero(), |s, j| s + a[k * n + j] * w[j]);
                    for j in (i + 1)..n {
                        a[k * n + j] = a[k * n + j] - tau * aw * w[j];
                    }
                }
                // V1 = V1 (I - tau w w^T): update the columns i+1..n.
                for i2 in 0..n {
                    let vw: F = ((i + 1)..n).fold(F::zero(), |s, j| s + v1[i2 * n + j] * w[j]);
                    if vw != F::zero() {
                        for j in (i + 1)..n {
                            v1[i2 * n + j] = v1[i2 * n + j] - tau * vw * w[j];
                        }
                    }
                }
            }
        }
    }
    (u1, a, v1)
}

/// LAPACK dbdsqr-style shifted QR iteration on an upper bidiagonal matrix
/// (main diagonal `d`, superdiagonal `e`). Returns the singular values and
/// logs the Givens rotations (k, cosl, sinl, cosr, sinr) in application
/// order so the caller can rebuild the singular vectors.
fn dbdsqr<F: Float + Copy>(
    d: &mut [F],
    e: &mut [F],
    givens: &mut Vec<(usize, F, F, F, F)>,
    max_sweeps: usize,
) -> Vec<F> {
    let n = d.len();
    let eps = F::epsilon();
    let tol = eps * F::from(10.0).unwrap();
    let mut smax = F::zero();
    for &x in d.iter().chain(e.iter()) {
        smax = smax.max(x.abs());
    }
    if smax == F::zero() {
        return d.iter().map(|x| x.abs()).collect();
    }
    // Perturb exact zeros so the sweeps never divide by zero.
    let floor = eps * smax;
    for (i, x) in d.iter_mut().enumerate() {
        if *x == F::zero() {
            *x = if i % 2 == 0 { floor } else { -floor };
        }
    }
    for (i, x) in e.iter_mut().enumerate() {
        if *x == F::zero() {
            *x = if i % 2 == 0 { floor } else { -floor };
        }
    }
    let mut m = n;
    let mut iters = 0;
    while m > 1 {
        // Find the lowest split: the block is [ll..m).
        let mut ll = 0;
        for k in (0..m - 1).rev() {
            if e[k].abs() <= tol * d[k].abs().max(d[k + 1].abs()) {
                e[k] = F::zero();
                ll = k + 1;
                break;
            }
        }
        if m - ll == 1 {
            m = ll;
            continue;
        }
        // Wilkinson-style shift from the bottom 2x2 block of B^T B.
        let shift = dlas2_smax(d[m - 2], e[m - 2], d[m - 1]);
        // One QR sweep over the block.
        let mut f = (d[ll].abs() - shift) * (d[ll].signum() + shift / d[ll]);
        let mut g = e[ll];
        for i in ll..m - 1 {
            let (cr, sr, r) = dlartg(f, g);
            if i > ll {
                e[i - 1] = r;
            }
            f = cr * d[i] + sr * e[i];
            e[i] = cr * e[i] - sr * d[i];
            g = sr * d[i + 1];
            d[i + 1] = cr * d[i + 1];
            let (cl, sl, r) = dlartg(f, g);
            d[i] = r;
            f = cl * e[i] + sl * d[i + 1];
            d[i + 1] = cl * d[i + 1] - sl * e[i];
            if i < m - 2 {
                g = sl * e[i + 1];
                e[i + 1] = cl * e[i + 1];
            }
            givens.push((i, cl, sl, cr, sr));
        }
        e[m - 2] = f;
        iters += 1;
        if iters > max_sweeps * n {
            break;
        }
    }
    d.iter().map(|x| x.abs()).collect()
}

/// Largest singular value of the 2x2 block [[d1, e1], [0, d2]].
fn dlas2_smax<F: Float + Copy>(d1: F, e1: F, d2: F) -> F {
    let t = d1 * d1 + d2 * d2 + e1 * e1;
    let disc = (t * t - F::from(4.0).unwrap() * d1 * d1 * d2 * d2)
        .max(F::zero())
        .sqrt();
    ((t + disc) / F::from(2.0).unwrap()).max(F::zero()).sqrt()
}

/// Givens rotation annihilating g: (f, g) -> (r, 0).
fn dlartg<F: Float + Copy>(f: F, g: F) -> (F, F, F) {
    let r = (f * f + g * g).sqrt();
    if r == F::zero() {
        (F::one(), F::zero(), F::zero())
    } else {
        (f / r, g / r, r)
    }
}
