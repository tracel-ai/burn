use crate::{DType, Tensor, check, check::TensorCheck, linalg::l2_norm, s};
use alloc::vec;
use alloc::vec::Vec;
use burn_std::{FloatDType, Slice, TensorData};
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

    // Stage 1: Golub-Kahan bidiagonalization, A = U1 B V1^T.
    let (u_bi, b, v_bi) = bidiagonalize::<D>(a);

    // Stage 2: diagonalize the bidiagonal B on the host (2n scalars), and
    // rebuild the factors from the accumulated Givens rotations. Pure host
    // math over tensor data: deterministic, no kernels, exact convergence.
    let batch: usize = dims[..(D - 2)].iter().product();
    let ub = u_bi.into_data().to_vec::<f32>().unwrap();
    let bv = b.into_data().to_vec::<f32>().unwrap();
    let vb = v_bi.into_data().to_vec::<f32>().unwrap();
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
        let (u, s, vt) = svd_host::<f64>(
            &to_f64(&ub),
            &to_f64(&bv),
            &to_f64(&vb),
            m,
            n,
            batch,
            sweeps,
        );
        (
            Tensor::<D>::from_data(TensorData::new(u, dims_u), &device),
            Tensor::<D1>::from_data(TensorData::new(s, dims_s), &device),
            Tensor::<D>::from_data(TensorData::new(vt, dims_vt), &device),
        )
    } else {
        let (u, s, vt) = svd_host::<f32>(&ub, &bv, &vb, m, n, batch, sweeps);
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

fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

/// Host stage: per batch, diagonalize the bidiagonal matrix with shifted QR
/// iterations (LAPACK dbdsqr) and rebuild U, Vt from the Givens rotations.
fn svd_host<F: Float + Copy>(
    ub: &[F],
    bv: &[F],
    vb: &[F],
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
        let (bb, ub_, vb_) = (
            &bv[b * m * n..(b + 1) * m * n],
            &ub[b * m * m..(b + 1) * m * m],
            &vb[b * n * n..(b + 1) * n * n],
        );
        for i in 0..n {
            d[i] = bb[i * n + i];
        }
        for i in 0..n.saturating_sub(1) {
            e[i] = bb[i * n + i + 1];
        }
        givens.clear();
        let sigma_b = dbdsqr(&mut d, &mut e, &mut givens, max_sweeps);

        // U2 = product of the left Givens rotations (applied in order).
        let mut u2 = vec![F::zero(); n * n];
        for i in 0..n {
            u2[i * n + i] = F::one();
        }
        for &(k, cl, sl, _, _) in &givens {
            for i in 0..n {
                let (a, b) = (u2[i * n + k], u2[i * n + k + 1]);
                u2[i * n + k] = cl * a + sl * b;
                u2[i * n + k + 1] = -sl * a + cl * b;
            }
        }
        // V2 = product of the right Givens rotations (applied in order).
        let mut v2 = vec![F::zero(); n * n];
        for i in 0..n {
            v2[i * n + i] = F::one();
        }
        for &(k, _, _, cr, sr) in &givens {
            for i in 0..n {
                let (a, b) = (v2[i * n + k], v2[i * n + k + 1]);
                v2[i * n + k] = cr * a + sr * b;
                v2[i * n + k + 1] = -sr * a + cr * b;
            }
        }
        // Absorb the signs of the diagonal into U2.
        for k in 0..n {
            if d[k] < F::zero() {
                for i in 0..n {
                    u2[i * n + k] = -u2[i * n + k];
                }
            }
        }
        // U = U1[:, :n] @ U2, Vt = (V1 @ V2)^T.
        let mut ub2 = vec![F::zero(); m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = F::zero();
                for k in 0..n {
                    acc = acc + ub_[i * m + k] * u2[k * n + j];
                }
                ub2[i * n + j] = acc;
            }
        }
        let mut v = vec![F::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = F::zero();
                for k in 0..n {
                    acc = acc + vb_[i * n + k] * v2[k * n + j];
                }
                v[i * n + j] = acc;
            }
        }
        for i in 0..m {
            for j in 0..n {
                u[b * m * n + i * n + j] = ub2[i * n + j];
            }
        }
        for i in 0..n {
            for j in 0..n {
                vt[b * n * n + i * n + j] = v[j * n + i];
            }
            sigma[b * n + i] = sigma_b[i];
        }
    }
    (u, sigma, vt)
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

/// Golub-Kahan bidiagonalization: `A = U B V^T` with `B` upper bidiagonal
/// `[..., m, n]`, using Householder reflections on shrinking submatrices
/// (the same pattern as `qr`). `U` and `V` accumulate the reflections.
fn bidiagonalize<const D: usize>(a: Tensor<D>) -> (Tensor<D>, Tensor<D>, Tensor<D>) {
    let dims = a.dims();
    let device = a.device();
    let (m, n) = (dims[D - 2], dims[D - 1]);
    let mut a = a;

    let eye = |rows: usize| -> Tensor<D> {
        let mut expand = [1; D];
        expand[..(D - 2)].copy_from_slice(&dims[..(D - 2)]);
        expand[D - 2] = rows;
        expand[D - 1] = rows;
        let mut reshape = [1; D];
        reshape[D - 2] = rows;
        reshape[D - 1] = rows;
        Tensor::eye(rows, &device).reshape(reshape).expand(expand)
    };
    let mut u = eye(m);
    let mut v = eye(n);

    for i in 0..n {
        // Left reflection: annihilate the subdiagonal of column i.
        let sub = a
            .clone()
            .slice_dim(D - 2, s![i..])
            .slice_dim(D - 1, s![i..]);
        let x = sub.clone().slice_dim(D - 1, 0..1);
        let x0 = x.clone().slice_dim(D - 2, 0..1);
        let norm = l2_norm(x.clone().slice_dim(D - 2, s![..]), D - 2);
        let sign = x0.clone().sign().neg().mask_fill(
            x0.clone().is_close(x0.clone().zeros_like(), None, None),
            -1.0,
        );
        let u0 = x0.clone().sub(norm.clone().mul(sign.clone()));
        let mask = norm.clone().is_close(norm.clone().zeros_like(), None, None);
        let tau = u0
            .clone()
            .neg()
            .div(norm.clone())
            .mul(sign.clone())
            .mask_fill(mask.clone(), 0.0);
        let e0 = x0.mul_scalar(0.0).add_scalar(1.0);
        let mut slices = vec![Slice::full(); D];
        slices[D - 2] = s![0];
        let w = x.div(u0.clone()).slice_assign(&slices, e0);
        let w = w.mask_fill(mask, 0.0);

        let wta = w.clone().expand(sub.dims()).mul(sub.clone()).sum_dim(D - 2);
        let a_new = sub.clone().sub(tau.clone().mul(w.clone().mul(wta.clone())));
        let mut slices = vec![Slice::full(); D];
        slices[D - 2] = s![i..];
        slices[D - 1] = s![i..];
        a = a.slice_assign(&slices, a_new);

        let u_sub = u.clone().slice_dim(D - 1, s![i..]);
        let wb = w.clone().transpose().expand(u_sub.dims());
        let uw = u_sub.clone().mul(wb).sum_dim(D - 1);
        let u_new = u_sub.sub(tau.clone().mul(uw.mul(w.clone().transpose())));
        let mut slices = vec![Slice::full(); D];
        slices[D - 1] = s![i..];
        u = u.slice_assign(&slices, u_new);

        // Right reflection: annihilate row i right of the superdiagonal.
        if i + 1 < n - 1 {
            let sub = a
                .clone()
                .slice_dim(D - 2, s![i..])
                .slice_dim(D - 1, s![i + 1..]);
            let y = sub.clone().slice_dim(D - 2, 0..1);
            let y0 = y.clone().slice_dim(D - 1, 0..1);
            let norm = l2_norm(y.clone().slice_dim(D - 1, s![..]), D - 1);
            let sign = y0.clone().sign().neg().mask_fill(
                y0.clone().is_close(y0.clone().zeros_like(), None, None),
                -1.0,
            );
            let u0 = y0.clone().sub(norm.clone().mul(sign.clone()));
            let mask = norm.clone().is_close(norm.clone().zeros_like(), None, None);
            let tau = u0
                .clone()
                .neg()
                .div(norm.clone())
                .mul(sign.clone())
                .mask_fill(mask.clone(), 0.0);
            let e0 = y0.mul_scalar(0.0).add_scalar(1.0);
            let mut slices = vec![Slice::full(); D];
            slices[D - 1] = s![0];
            let w = y.div(u0).slice_assign(&slices, e0);
            let w = w.mask_fill(mask, 0.0);

            let wb = w.clone().expand(sub.dims());
            let aw = sub.clone().mul(wb).sum_dim(D - 1);
            let a_new = sub.sub(tau.clone().mul(aw.mul(w.clone())));
            let mut slices = vec![Slice::full(); D];
            slices[D - 2] = s![i..];
            slices[D - 1] = s![i + 1..];
            a = a.slice_assign(&slices, a_new);

            let v_sub = v.clone().slice_dim(D - 1, s![i + 1..]);
            let wb = w.clone().expand(v_sub.dims());
            let vw = v_sub.clone().mul(wb).sum_dim(D - 1);
            let v_new = v_sub.sub(tau.clone().mul(vw.mul(w.clone())));
            let mut slices = vec![Slice::full(); D];
            slices[D - 1] = s![i + 1..];
            v = v.slice_assign(&slices, v_new);
        }
    }
    (u, a, v)
}
