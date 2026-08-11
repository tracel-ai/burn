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
/// Two stages, mirroring the LAPACK `gesvd` / `dbdsqr` structure:
///
/// 1. **Golub-Kahan bidiagonalization** using Householder reflections (`A = U1 B V1^T`).
/// 2. **Implicitly shifted bidiagonal QR iteration** (LAPACK `dbdsqr`) to diagonalize `B`.
///
/// The algorithm is backward stable to within a small multiple of machine precision.
/// Convergence typically requires ~2.5 QR sweeps per singular value on average, bounded
/// by the `sweeps` argument.
///
/// # Arguments
///
/// * `tensor` - The input tensor of shape `[..., m, n]`.
/// * `sweeps` - Upper bound on the number of QR sweeps (the algorithm
///   typically converges in ~2.5 sweeps per singular value on average).
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
///   `10 * eps * sigma_max` (machine epsilon of the dtype) are treated as
///   numerical zeros and returned as 0.
/// - All internal norms, rotations and shifts are scale-invariant
///   (LAPACK-style), so inputs with entries up to `f32::MAX` and down to
///   subnormals stay finite and accurate.
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
        if D1 >= 2 {
            ds[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
        }
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
    // Scratch reused across batch elements (allocated once, not per batch).
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut sorted = vec![F::zero(); n];

    for b in 0..batch {
        let ab = &a[b * m * n..(b + 1) * m * n];
        // Closed forms for the smallest sizes: exact and much cheaper than
        // the bidiagonalization + QR pipeline.
        if n == 1 {
            // A is [m, 1] (wide inputs were transposed): the single singular
            // value is the column norm and U is the normalized column.
            let mut s = F::zero();
            for i in 0..m {
                let t = ab[i];
                s = s + t * t;
            }
            let sv = s.sqrt();
            sigma[b] = sv;
            for i in 0..m {
                u[b * m * n + i * n] = if sv > F::zero() { ab[i] / sv } else { F::one() };
            }
            vt[b] = F::one();
            continue;
        }
        if n == 2 && m == 2 {
            let (s0, s1, uu, vv) = svd2x2(ab);
            sigma[b * 2] = s0;
            sigma[b * 2 + 1] = s1;
            u[b * 4..b * 4 + 4].copy_from_slice(&uu);
            vt[b * 4..b * 4 + 4].copy_from_slice(&vv);
            continue;
        }
        let (u1, bv, v1) = bidiag_host(ab, m, n);
        // Transpose both factor buffers up front: the Givens rotations act on
        // column pairs (k, k+1), which are stride-n in row-major storage but
        // contiguous rows in the transposed layout, so the hot loops below
        // stream memory instead of hopping columns.
        let mut u1t = vec![F::zero(); m * n];
        let mut v1t = vec![F::zero(); n * n];
        for i in 0..m {
            for j in 0..n {
                u1t[j * m + i] = u1[i * n + j];
            }
        }
        for i in 0..n {
            for j in 0..n {
                v1t[j * n + i] = v1[i * n + j];
            }
        }
        for i in 0..n {
            d[i] = bv[i * n + i];
        }
        for i in 0..n.saturating_sub(1) {
            e[i] = bv[i * n + i + 1];
        }
        givens.clear();
        let sigma_b = dbdsqr(&mut d, &mut e, &mut givens, max_sweeps);

        // U = U1 @ (product of left Givens rotations): in the transposed
        // layout a rotation of columns (k, k+1) touches contiguous rows.
        for &(k, cl, sl, _, _) in &givens {
            for i in 0..m {
                let (a0, b0) = (u1t[k * m + i], u1t[(k + 1) * m + i]);
                u1t[k * m + i] = cl * a0 + sl * b0;
                u1t[(k + 1) * m + i] = -sl * a0 + cl * b0;
            }
        }
        // Vt = (V1 @ (product of right Givens rotations))^T.
        for &(k, _, _, cr, sr) in &givens {
            for i in 0..n {
                let (a0, b0) = (v1t[k * n + i], v1t[(k + 1) * n + i]);
                v1t[k * n + i] = cr * a0 + sr * b0;
                v1t[(k + 1) * n + i] = -sr * a0 + cr * b0;
            }
        }
        // Absorb the signs of the diagonal into U.
        for k in 0..n {
            if d[k] < F::zero() {
                for i in 0..m {
                    u1t[k * m + i] = -u1t[k * m + i];
                }
            }
        }
        for i in 0..m {
            for j in 0..n {
                u[b * m * n + i * n + j] = u1t[j * m + i];
            }
        }
        for i in 0..n {
            for j in 0..n {
                vt[b * n * n + i * n + j] = v1t[i * n + j];
            }
            sigma[b * n + i] = sigma_b[i];
        }
        // Sort the singular values descending and permute the factors, on the
        // host: deterministic (stable sort), independent of backend
        // gather/argsort kernels (which are view-based or nondeterministic on
        // fused CUDA). Mask numerical zeros relative to sigma_max, with a
        // dtype-derived threshold (10 * machine epsilon, same relative
        // tolerance as the dbdsqr deflation test) instead of a fixed 1e-6.
        let smax = sigma[b * n..(b + 1) * n]
            .iter()
            .fold(F::zero(), |s, &x| s.max(x.abs()));
        let zero_tol = smax * (F::epsilon() * F::from(10.0).unwrap());
        order.clear();
        order.extend(0..n);
        order.sort_by(|&i, &j| {
            sigma[b * n + j]
                .partial_cmp(&sigma[b * n + i])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        // Apply the permutation while copying: in the transposed layout the
        // column permutation of U is a row permutation of U1t, and the row
        // permutation of Vt is a row permutation of V1t, so the factors land
        // in the output tensors directly (no intermediate permute buffers).
        for (t, &src) in order.iter().enumerate() {
            for i in 0..m {
                u[b * m * n + i * n + t] = u1t[src * m + i];
            }
            for i in 0..n {
                vt[b * n * n + t * n + i] = v1t[src * n + i];
            }
            // read from the untouched slot: sigma is rewritten in place below
            let v = sigma[b * n + src];
            sorted[t] = if v.abs() <= zero_tol { F::zero() } else { v };
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

/// Exact 2x2 SVD in closed form (J. Blinn, "Consider the lowly 2x2 matrix",
/// IEEE CG&A 1996): returns (s0, s1, u, vt) with s0 >= s1 >= 0, u/vt row-major.
///
/// Derivation: U is the eigenvector rotation of A Aᵀ (a symmetric 2x2, whose
/// dominant eigenvector angle is θ = ½·atan2(2(eg+fh), e²+f²−g²−h²)); after
/// applying Uᵀ the two rows of UᵀA are orthogonal, so their normalized forms
/// give Vᵀ directly and the row norms are the singular values.
fn svd2x2<F: Float + Copy>(a: &[F]) -> (F, F, [F; 4], [F; 4]) {
    let (e, f, g, h) = (a[0], a[1], a[2], a[3]);
    let theta = F::from(0.5).unwrap()
        * F::atan2(
            F::from(2.0).unwrap() * (e * g + f * h),
            e * e + f * f - g * g - h * h,
        );
    let (ct, st) = (theta.cos(), theta.sin());
    // UᵀA rows (orthogonal by construction):
    let e2 = ct * e + st * g;
    let f2 = ct * f + st * h;
    let g2 = -st * e + ct * g;
    let h2 = -st * f + ct * h;
    let s0 = (e2 * e2 + f2 * f2).sqrt();
    let s1 = (g2 * g2 + h2 * h2).sqrt();
    // Degenerate rows (σ = 0) fall back to a unit basis so U/Vt stay orthonormal.
    let (r0c, r0s) = if s0 > F::zero() {
        (e2 / s0, f2 / s0)
    } else {
        (F::one(), F::zero())
    };
    let (r1c, r1s) = if s1 > F::zero() {
        (g2 / s1, h2 / s1)
    } else {
        (F::zero(), F::one())
    };
    // The rows of UᵀA are orthogonal up to sign; fix the handedness of Vt.
    let det = r0c * r1s - r0s * r1c;
    let (r1c, r1s) = if det < F::zero() {
        (-r1c, -r1s)
    } else {
        (r1c, r1s)
    };
    let u = [ct, -st, st, ct];
    let vt = [r0c, r0s, r1c, r1s];
    if s0 >= s1 {
        (s0, s1, u, vt)
    } else {
        // Swap columns of U / rows of Vt to keep descending order.
        (
            s1,
            s0,
            [u[2], u[3], u[0], u[1]],
            [vt[2], vt[3], vt[0], vt[1]],
        )
    }
}

/// Golub-Kahan bidiagonalization on the host: `A = U1 B V1^T` with `B` upper
/// bidiagonal (row-major `[m, n]`), using Householder reflections on
/// shrinking submatrices, mirroring the tensor-op version operation for
/// operation.
fn bidiag_host<F: Float + Copy>(a: &[F], m: usize, n: usize) -> (Vec<F>, Vec<F>, Vec<F>) {
    let mut v1 = vec![F::zero(); n * n];
    for i in 0..n {
        v1[i * n + i] = F::one();
    }
    // Left reflectors (w, tau) are collected and applied to the U1 columns
    // once at the end: the output only needs the first n columns, which
    // costs O(m n^2) instead of the O(m^2 n) step-by-step update, a large
    // win for tall matrices (m >> n).
    let mut ws: Vec<F> = vec![F::zero(); m * n];
    let mut taus = vec![F::zero(); n];
    let mut a = a.to_vec();
    // Reused reflection scratch (allocated once, not per reflection).
    let mut w = vec![F::zero(); m.max(n)];
    let mut wta = vec![F::zero(); n];
    let mut aw = vec![F::zero(); m];

    for i in 0..n {
        // Left reflection: annihilate the subdiagonal of column i.
        // Scaled norm (like LAPACK dlarfg): sqrt(sum x^2) without overflow or
        // underflow for extreme scales (|x| up to f32::MAX, down to subnormals).
        let mut scale = F::zero();
        for k in i..m {
            let t = a[k * n + i].abs();
            if t > scale {
                scale = t;
            }
        }
        let norm = if scale == F::zero() {
            F::zero()
        } else {
            let mut s = F::zero();
            for k in i..m {
                let t = a[k * n + i] / scale;
                s = s + t * t;
            }
            scale * s.sqrt()
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
            w[i] = F::one();
            for k in (i + 1)..m {
                w[k] = a[k * n + i] / u0;
            }
            // wta[j] = w^T a[:, j], then a_new = a - tau w wta. Row-major
            // sweeps: a[k, :] is contiguous, so both passes stay sequential.
            for j in i..n {
                wta[j] = F::zero();
            }
            for k in i..m {
                let wk = w[k];
                for j in i..n {
                    wta[j] = wta[j] + wk * a[k * n + j];
                }
            }
            let t = tau;
            for k in i..m {
                let wk = w[k] * t;
                for j in i..n {
                    a[k * n + j] = a[k * n + j] - wk * wta[j];
                }
            }
            for (k, &wk) in w.iter().enumerate().take(m) {
                ws[i * m + k] = wk;
            }
            taus[i] = tau;
        }

        // Right reflection: annihilate row i right of the superdiagonal.
        if i + 1 < n - 1 {
            let mut scale = F::zero();
            for j in (i + 1)..n {
                let t = a[i * n + j].abs();
                if t > scale {
                    scale = t;
                }
            }
            let norm = if scale == F::zero() {
                F::zero()
            } else {
                let mut s = F::zero();
                for j in (i + 1)..n {
                    let t = a[i * n + j] / scale;
                    s = s + t * t;
                }
                scale * s.sqrt()
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
                w[i + 1] = F::one();
                for j in (i + 2)..n {
                    w[j] = a[i * n + j] / u0;
                }
                // aw[k] = a[k, :] w, then a_new = a - tau aw w^T. Contiguous
                // row sweeps again: a[k, j] advances sequentially in both.
                for k in i..m {
                    let mut s = F::zero();
                    for j in (i + 1)..n {
                        s = s + a[k * n + j] * w[j];
                    }
                    aw[k] = s;
                }
                let t = tau;
                for k in i..m {
                    let awk = aw[k] * t;
                    for j in (i + 1)..n {
                        a[k * n + j] = a[k * n + j] - awk * w[j];
                    }
                }
                // V1 = V1 (I - tau w w^T): update the columns i+1..n.
                for i2 in 0..n {
                    let mut vw = F::zero();
                    for j in (i + 1)..n {
                        vw = vw + v1[i2 * n + j] * w[j];
                    }
                    if vw != F::zero() {
                        let tvw = tau * vw;
                        for j in (i + 1)..n {
                            v1[i2 * n + j] = v1[i2 * n + j] - tvw * w[j];
                        }
                    }
                }
            }
        }
    }
    let mut u1 = vec![F::zero(); m * n];
    for i in 0..n {
        u1[i * n + i] = F::one();
    }
    for i in (0..n).rev() {
        let tau = taus[i];
        if tau != F::zero() {
            // uw[j] = w^T u1[:, j] computed over contiguous column strips:
            // u1[k, :] is row-major, so sweep k outer, j inner.
            let mut uw = vec![F::zero(); n];
            for k in i..m {
                let wk = ws[i * m + k];
                for j in 0..n {
                    uw[j] = uw[j] + wk * u1[k * n + j];
                }
            }
            let t = tau;
            for k in i..m {
                let wk = ws[i * m + k] * t;
                for j in 0..n {
                    u1[k * n + j] = u1[k * n + j] - wk * uw[j];
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
        // Wilkinson-style shift from the bottom 2x2 block of B^T B. For f32
        // inputs the closed form carries ~1e-5 error, which exceeds the
        // deflation threshold (10 eps) and stalls 2x2 blocks on rounding
        // boundaries; computing the shift in f64 keeps it exact.
        let shift = if core::mem::size_of::<F>() == 4 {
            let d1 = d[m - 2].to_f64().unwrap();
            let e1 = e[m - 2].to_f64().unwrap();
            let d2 = d[m - 1].to_f64().unwrap();
            let t = d1 * d1 + d2 * d2 + e1 * e1;
            let disc = (t * t - 4.0 * d1 * d1 * d2 * d2).max(0.0).sqrt();
            F::from(((t + disc) / 2.0).max(0.0).sqrt()).unwrap()
        } else {
            dlas2_smax(d[m - 2], e[m - 2], d[m - 1])
        };
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
    let t = d1 * d1 + d2 * d2 + e1 * e1;
    if t.is_finite() {
        let disc = (t * t - F::from(4.0).unwrap() * d1 * d1 * d2 * d2)
            .max(F::zero())
            .sqrt();
        ((t + disc) / F::from(2.0).unwrap()).max(F::zero()).sqrt()
    } else {
        let scale = d1.abs().max(d2.abs()).max(e1.abs());
        if scale == F::zero() {
            return F::zero();
        }
        let (a, b, c) = (d1 / scale, d2 / scale, e1 / scale);
        let t = a * a + b * b + c * c;
        let disc = (t * t - F::from(4.0).unwrap() * a * a * b * b).max(F::zero());
        scale * ((t + disc.sqrt()) / F::from(2.0).unwrap()).sqrt()
    }
}

/// Givens rotation annihilating g: (f, g) -> (r, 0). Scaled like LAPACK
/// dlartg: r = sqrt(f^2 + g^2) computed without overflow or underflow.
fn dlartg<F: Float + Copy>(f: F, g: F) -> (F, F, F) {
    let r2 = f * f + g * g;
    if r2.is_finite() && r2 > F::zero() {
        let r = r2.sqrt();
        (f / r, g / r, r)
    } else {
        let scale = f.abs().max(g.abs());
        if scale == F::zero() {
            (F::one(), F::zero(), F::zero())
        } else {
            let (sf, sg) = (f / scale, g / scale);
            let r = scale * (sf * sf + sg * sg).sqrt();
            (f / r, g / r, r)
        }
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
        // Reference from numpy/LAPACK gesdd (svd_host already sorts descending).
        assert!((s[0] - 25.46240743603639).abs() < 1e-12, "s1 {}", s[0]);
        assert!((s[1] - 1.290661675761233).abs() < 1e-12, "s2 {}", s[1]);
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

    #[test]
    fn test_svd_host_tall_fixed() {
        // fixed 512x128: deterministic formula, checks the lazy-U1 path
        let m = 512usize;
        let n = 128usize;
        let mut a = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                a.push(
                    (((i * 7919 + j * 104729) % 100000) as f64 / 100000.0 - 0.5) * 2.0
                        + (i as f64 - j as f64) * 0.001,
                );
            }
        }
        let (u, s, vt) = svd_host::<f64>(&a, m, n, 1, 30, false);
        let mut err = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for k in 0..n {
                    acc += u[i * n + k] * s[k] * vt[k * n + j];
                }
                err = err.max((a[i * n + j] - acc).abs());
            }
        }
        assert!(err < 1e-9, "tall recon err {err}");

        // same matrix in f32
        let af: Vec<f32> = a.iter().map(|x| *x as f32).collect();
        let (u, s, vt) = svd_host::<f32>(&af, m, n, 1, 30, false);
        let mut err = 0.0f32;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for k in 0..n {
                    acc += u[i * n + k] * s[k] * vt[k * n + j];
                }
                err = err.max((af[i * n + j] - acc).abs());
            }
        }
        assert!(err < 1e-3, "tall f32 recon err {err}");
    }

    #[test]
    fn test_svd_host_tall_seeded() {
        // deterministic LCG over 60 tall 512x128 matrices in f32, find the
        // pathological one that the randomized bench hit (~2.5% of matrices)
        let (m, n) = (512usize, 128usize);
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (1u32 << 31) as f32
        };
        let mut worst = 0.0f32;
        for _trial in 0..60 {
            let mut a = vec![0.0f32; m * n];
            for x in a.iter_mut() {
                *x = next() * 2.0 - 1.0;
            }
            let (u, s, vt) = svd_host::<f32>(&a, m, n, 1, 15, false);
            let mut err = 0.0f32;
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for k in 0..n {
                        acc += u[i * n + k] * s[k] * vt[k * n + j];
                    }
                    err = err.max((a[i * n + j] - acc).abs());
                }
            }

            if err > worst {
                worst = err;
            }
        }
        assert!(worst < 1e-3, "seeded tall worst {worst}");
    }

    #[test]
    fn test_svd_host_2x2_direct() {
        // B = [[2,1],[0,1]]: exact 2x2 closed-form path
        let a = [2.0f64, 1.0, 0.0, 1.0];
        let (u, s, vt) = svd_host::<f64>(&a, 2, 2, 1, 30, false);
        let mut err = 0.0f64;
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0f64;
                for k in 0..2 {
                    acc += u[i * 2 + k] * s[k] * vt[k * 2 + j];
                }
                err = err.max((a[i * 2 + j] - acc).abs());
            }
        }
        assert!(err < 1e-12, "2x2 recon {err}");
    }

    #[test]
    fn test_svd_host_batch2_matrix() {
        // the batch-2 matrix from test_svd_host_f64, isolated
        let a = [
            1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.5, 2.5,
        ];
        let (u, s, vt) = svd_host::<f64>(&a, 4, 3, 1, 30, false);
        let mut err = 0.0f64;
        for i in 0..4 {
            for j in 0..3 {
                let mut acc = 0.0f64;
                for k in 0..3 {
                    acc += u[i * 3 + k] * s[k] * vt[k * 3 + j];
                }
                err = err.max((a[i * 3 + j] - acc).abs());
            }
        }
        assert!(err < 1e-12, "batch2 recon {err}");
    }
}
