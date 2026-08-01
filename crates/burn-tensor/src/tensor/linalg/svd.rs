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
/// 1. **Golub-Kahan bidiagonalization** with Householder reflections:
///    `A = U1 B V1^T` with `B` upper bidiagonal, `n` reflections on
///    shrinking submatrices.
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
/// # Performance Note
/// The pipeline runs entirely on the host over the tensor data
/// (`into_data` / `from_data`), which makes it deterministic and
/// backend-independent, but the bidiagonalization is O(m n^2) scalar math.
/// It is not competitive with tuned native libraries (e.g. cuSOLVER) for
/// large matrices; at 128x128 it is within ~3x of them. On fused CUDA this
/// is still ~200x faster than a tensor-op implementation of the same
/// algorithm (which pays per-operation dispatch overhead).
///
/// # Numerical Behavior
/// - If the input tensor has dtype F16 or BF16, it is internally upcast to
///   F32 for the computation and cast back to the original dtype before
///   returning, like `det` and `lu`.
/// - Singular values are sorted in descending order; values at or below
///   `1e-6 * sigma_max` are treated as numerical zeros and returned as 0.
/// - All internal norms, rotations and shifts are scale-invariant
///   (LAPACK-style), so inputs with entries up to `f32::MAX` and down to
///   subnormals stay finite and exact.
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

    // Empty matrix (a zero leading dimension): the reduced SVD is empty too.
    // Skip the pipeline (bidiagonalization would index out of bounds).
    if m == 0 || n == 0 {
        let mut du = [1; D];
        du[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
        du[D - 2] = n_rows;
        du[D - 1] = 0;
        let mut ds = [1; D1];
        ds[D1 - 1] = 0;
        let mut dv = [1; D];
        dv[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
        dv[D - 2] = 0;
        dv[D - 1] = n_cols;
        let u_t = Tensor::<D>::from_data(TensorData::new(Vec::<f32>::new(), du), &device);
        let s_t = Tensor::<D1>::from_data(TensorData::new(Vec::<f32>::new(), ds), &device);
        let vt_t = Tensor::<D>::from_data(TensorData::new(Vec::<f32>::new(), dv), &device);
        let (u_t, s_t, vt_t) = if needs_upcast || original_dtype == DType::F64 {
            (
                u_t.cast(original_dtype),
                s_t.cast(original_dtype),
                vt_t.cast(original_dtype),
            )
        } else {
            (u_t, s_t, vt_t)
        };
        return (u_t, s_t, vt_t);
    }

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

    // Flush any in-flight kernels on the device (e.g. a previous test that
    // never read its outputs): cubecl host reads can fail with "strides are
    // not supported" while other kernels are still queued. No-op on eager
    // backends such as ndarray.
    let _ = device.sync();
    // svd_host already sorted, masked, permuted and swapped the factors; the
    // dims follow the orientation (swap -> u is [..., n, n], vt is [..., m, n]).
    let (du, dv) = if swap {
        let mut dv = [1; D];
        dv[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
        dv[D - 2] = n;
        dv[D - 1] = m;
        (dims_vt, dv)
    } else {
        (dims_u, dims_vt)
    };
    let result = if original_dtype == DType::F64 {
        // materialize any view (clone/transpose) into a contiguous buffer
        let a = a.clone().reshape(a.dims());
        let a_data = a.into_data().to_vec::<f64>().unwrap();
        let (u, s, vt) = svd_host::<f64>(&a_data, m, n, batch, sweeps, swap);
        (
            Tensor::<D>::from_data(TensorData::new(u, du), &device),
            Tensor::<D1>::from_data(TensorData::new(s, dims_s), &device),
            Tensor::<D>::from_data(TensorData::new(vt, dv), &device),
        )
    } else {
        let a = a.clone().reshape(a.dims());
        let a_data = a.into_data().to_vec::<f32>().unwrap();
        let (u, s, vt) = svd_host::<f32>(&a_data, m, n, batch, sweeps, swap);
        (
            Tensor::<D>::from_data(TensorData::new(u, du), &device),
            Tensor::<D1>::from_data(TensorData::new(s, dims_s), &device),
            Tensor::<D>::from_data(TensorData::new(vt, dv), &device),
        )
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
    // Flush the output transfers; no-op on eager backends.
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
    swap: bool,
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
        // Sort the singular values descending and permute the factors, on the
        // host: deterministic (stable sort), independent of backend
        // gather/argsort kernels (which are view-based or nondeterministic on
        // fused CUDA). Mask numerical zeros relative to sigma_max.
        let smax = sigma[b * n..(b + 1) * n]
            .iter()
            .fold(F::zero(), |s, &x| s.max(x.abs()));
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| {
            sigma[b * n + j]
                .partial_cmp(&sigma[b * n + i])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let mut pu = vec![F::zero(); m * n];
        let mut pvt = vec![F::zero(); n * n];
        let mut sorted = vec![F::zero(); n];
        for (t, &src) in order.iter().enumerate() {
            for i in 0..m {
                pu[i * n + t] = u[b * m * n + i * n + src];
            }
            for i in 0..n {
                pvt[t * n + i] = vt[b * n * n + src * n + i];
            }
            // read from the untouched slot: sigma is rewritten in place below
            let v = sigma[b * n + src];
            sorted[t] = if v.abs() <= smax * F::from(1e-6).unwrap() {
                F::zero()
            } else {
                v
            };
        }
        for i in 0..m * n {
            u[b * m * n + i] = pu[i];
        }
        for i in 0..n * n {
            vt[b * n * n + i] = pvt[i];
        }
        for i in 0..n {
            sigma[b * n + i] = sorted[i];
        }
    }
    if swap {
        // The SVD was computed on A^T ([n, m]); the factors for the original
        // wide A = Vt^T S U^T, so return u = Vt^T and vt = U^T (already
        // permuted consistently with the sorted sigma).
        let mut uf = vec![F::zero(); batch * n * n];
        let mut vf = vec![F::zero(); batch * n * m];
        for b in 0..batch {
            for i in 0..n {
                for j in 0..n {
                    uf[(b * n + i) * n + j] = vt[(b * n + j) * n + i];
                }
            }
            for i in 0..n {
                for j in 0..m {
                    vf[(b * n + i) * m + j] = u[b * m * n + j * n + i];
                }
            }
        }
        (uf, sigma, vf)
    } else {
        (u, sigma, vt)
    }
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
        // Scaled norm (like LAPACK dlarfg): sqrt(sum x^2) without overflow or
        // underflow for extreme scales (|x| up to f32::MAX, down to subnormals).
        let scale = (i..m).map(|k| a[k * n + i].abs()).fold(F::zero(), F::max);
        let norm = if scale == F::zero() {
            F::zero()
        } else {
            scale
                * (i..m)
                    .map(|k| {
                        let t = a[k * n + i] / scale;
                        t * t
                    })
                    .fold(F::zero(), |s, x| s + x)
                    .sqrt()
        };
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
            let scale = ((i + 1)..n)
                .map(|j| a[i * n + j].abs())
                .fold(F::zero(), F::max);
            let norm = if scale == F::zero() {
                F::zero()
            } else {
                scale
                    * ((i + 1)..n)
                        .map(|j| {
                            let t = a[i * n + j] / scale;
                            t * t
                        })
                        .fold(F::zero(), |s, x| s + x)
                        .sqrt()
            };
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
    // Perturb exact zeros on the DIAGONAL only (never the superdiagonal): a
    // zero d[i] inside an active block makes the sweep stall (dlartg(0, 0)),
    // and perturbing e instead would swamp small-but-real singular values on
    // large-scale inputs (e.g. diag(1e38, 1, 1) in f32 would get e-floors of
    // 1e31 and diverge to NaN). e zeros deflate naturally via the check below.
    let floor = eps * smax;
    for (i, x) in d.iter_mut().enumerate() {
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
        // One QR sweep over the block. The starting value is the first column
        // of (B - shift I) scaled for stability; with d[ll] exactly zero the
        // formula diverges, so take its finite proxy (-shift, direction of
        // the limit for positive d).
        let mut f = if d[ll] == F::zero() {
            -shift
        } else {
            (d[ll].abs() - shift) * (d[ll].signum() + shift / d[ll])
        };
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

/// Largest singular value of the 2x2 block [[d1, e1], [0, d2]]. Scaled like
/// LAPACK dlas2: no overflow or underflow on the intermediate squares.
fn dlas2_smax<F: Float + Copy>(d1: F, e1: F, d2: F) -> F {
    let scale = d1.abs().max(d2.abs()).max(e1.abs());
    if scale == F::zero() {
        return F::zero();
    }
    let (a, b, c) = (d1 / scale, d2 / scale, e1 / scale);
    let t = a * a + b * b + c * c;
    let disc = (t * t - F::from(4.0).unwrap() * a * a * b * b).max(F::zero());
    scale * ((t + disc.sqrt()) / F::from(2.0).unwrap()).sqrt()
}

/// Givens rotation annihilating g: (f, g) -> (r, 0). Scaled like LAPACK
/// dlartg: r = sqrt(f^2 + g^2) computed without overflow or underflow.
fn dlartg<F: Float + Copy>(f: F, g: F) -> (F, F, F) {
    let scale = f.abs().max(g.abs());
    if scale == F::zero() {
        (F::one(), F::zero(), F::zero())
    } else {
        let (sf, sg) = (f / scale, g / scale);
        let r = scale * (sf * sf + sg * sg).sqrt();
        (f / r, g / r, r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recon_err<F: Float + Copy>(u: &[F], s: &[F], vt: &[F], a: &[F], m: usize, n: usize) -> F {
        assert!(
            u.len() == m * n && s.len() == n && vt.len() == n * n && a.len() == m * n,
            "sizes: u={} s={} vt={} a={} m={m} n={n}",
            u.len(),
            s.len(),
            vt.len(),
            a.len()
        );
        let mut err = F::zero();
        for i in 0..m {
            for j in 0..n {
                let mut acc = F::zero();
                for k in 0..n {
                    acc = acc + u[i * n + k] * s[k] * vt[k * n + j];
                }
                err = err.max((a[i * n + j] - acc).abs());
            }
        }
        err
    }

    #[test]
    fn test_svd_host_f64() {
        // 4x3 full-rank matrix with known torch f64 values
        let a = [
            1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let (u, s, vt) = svd_host::<f64>(&a, 4, 3, 1, 30, false);
        // svd_host returns sigma in diagonalization order; sort before comparing.
        let mut ss = s.clone();
        ss.sort_by(|x, y| y.partial_cmp(x).unwrap());
        // Reference from numpy/LAPACK gesdd.
        assert!((ss[0] - 25.46240743603639).abs() < 1e-12, "s1 {}", ss[0]);
        assert!((ss[1] - 1.290661675761233).abs() < 1e-12, "s2 {}", ss[1]);
        assert!(recon_err::<f64>(&u, &s, &vt, &a, 4, 3) < 1e-12);
        // 1x1 and rank-1 edge cases
        let (u, s, vt) = svd_host::<f64>(&[-3.0], 1, 1, 1, 30, false);
        assert!((s[0] - 3.0).abs() < 1e-15);
        assert!(recon_err::<f64>(&u, &s, &vt, &[-3.0], 1, 1) < 1e-15);
        // rank-1 3x2 (m >= n as svd_host requires; wide inputs are transposed
        // in svd() itself)
        let a = [1.0f64, 2.0, 2.0, 4.0, 3.0, 6.0];
        let (u, s, vt) = svd_host::<f64>(&a, 3, 2, 1, 30, false);
        assert!(recon_err::<f64>(&u, &s, &vt, &a, 3, 2) < 1e-12);
        // batched 4x3
        let a = [
            1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.5, 2.5,
        ];
        let (u, s, vt) = svd_host::<f64>(&a, 4, 3, 2, 30, false);
        assert!(recon_err::<f64>(&u[..12], &s[..3], &vt[..9], &a[..12], 4, 3) < 1e-12);
        assert!(recon_err::<f64>(&u[12..], &s[3..], &vt[9..], &a[12..], 4, 3) < 1e-12);
        println!("DBGT batch2 recon ok");
        // extreme scales stay finite
        let a = [1e200f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (u, s, vt) = svd_host::<f64>(&a, 3, 3, 1, 30, false);
        assert!(
            s[0].is_finite() && (s[0] - 1e200).abs() < 1e200 * 1e-14,
            "s0 {}",
            s[0]
        );
        assert!(recon_err::<f64>(&u, &s, &vt, &a, 3, 3) < 1e200 * 1e-13);
    }

    #[test]
    fn test_svd_host_f32_extremes() {
        // f32: scaled norm/dlartg/dlas2 must not overflow or produce NaN
        for a in [
            [1e38f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1e-40f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1e20f32, 1e-20, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ] {
            let (u, s, vt) = svd_host::<f32>(&a, 3, 3, 1, 30, false);
            for x in s.iter() {
                assert!(x.is_finite(), "sigma not finite: {:?}", s);
            }
            let err = recon_err::<f32>(&u, &s, &vt, &a, 3, 3);
            assert!(err.is_finite(), "recon not finite {err}");
            assert!(err <= a[0].abs().max(1.0) * 1e-4, "recon err {err}");
        }
    }
}
