//! Host SVD pipeline: Golub-Kahan bidiagonalization + LAPACK-style `dbdsqr`
//! + Givens factor assembly, over plain slices.
//!
//! This is the reference implementation backing the default
//! [`LinalgOps::svd`](crate::LinalgOps::svd):
//! pure scalar math over tensor data, deterministic and identical on every
//! backend. Backends may override the trait method with a native SVD (tch)
//! or a fused GPU kernel (cubecl); this module stays the correctness
//! reference and the no-kernel fallback.
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use burn_std::{DType, ExecutionError, TensorData, backtrace::BackTrace};
use num_traits::float::Float;

type SvdFactors<F> = (Vec<F>, Vec<F>, Vec<F>);

/// Run the full host pipeline over tensor data and return the three factors
/// as data. Layout and dims follow the `swap` convention of
/// `crates/burn-tensor`'s `linalg::svd` (see `svd_host`).
///
/// Consumes the data so the input can be converted to a plain vector without
/// copying (the backend transfers are already owned by the caller).
/// Returns an execution error if any batch element exceeds the QR sweep
/// budget before converging.
pub(crate) fn svd_host_data(
    data: TensorData,
    sweeps: usize,
    swap: bool,
) -> Result<(TensorData, TensorData, TensorData), ExecutionError> {
    let rank = data.shape.num_dims();
    let dims: alloc::vec::Vec<usize> = data.shape.iter().copied().collect();
    let batch: usize = dims[..rank - 2].iter().product();
    let (m, n) = (dims[rank - 2], dims[rank - 1]);

    // Factor dims: without swap U is [.., m, k], Vt is [.., k, n]; with swap
    // (wide input) U is [.., k, k] and Vt is [.., k, m], k = n.
    let mut du = dims[..rank].to_vec();
    let mut ds = dims[..rank - 1].to_vec();
    let mut dv = dims[..rank].to_vec();
    du[rank - 1] = n;
    dv[rank - 2] = n;
    dv[rank - 1] = n;
    ds[rank - 2] = n;
    if swap {
        du[rank - 2] = n;
        dv[rank - 2] = n;
        dv[rank - 1] = m;
    }

    if data.dtype == DType::F64 {
        let a = data.try_into_vec::<f64>().unwrap();
        let (u, s, vt) = svd_host::<f64>(&a, m, n, batch, sweeps, swap)
            .map_err(|batch| convergence_error(batch, n, sweeps))?;
        Ok((
            TensorData::new(u, du),
            TensorData::new(s, ds),
            TensorData::new(vt, dv),
        ))
    } else {
        let a = data.try_into_vec::<f32>().unwrap();
        let (u, s, vt) = svd_host::<f32>(&a, m, n, batch, sweeps, swap)
            .map_err(|batch| convergence_error(batch, n, sweeps))?;
        Ok((
            TensorData::new(u, du),
            TensorData::new(s, ds),
            TensorData::new(vt, dv),
        ))
    }
}

fn convergence_error(batch: usize, n: usize, sweeps: usize) -> ExecutionError {
    let iterations = sweeps.saturating_mul(n);
    ExecutionError::Generic {
        reason: format!(
            "SVD QR iteration did not converge for batch element {batch} after {iterations} \
             iterations ({sweeps} sweeps per singular value)"
        ),
        backtrace: BackTrace::capture(),
    }
}

/// Host pipeline over a batch of matrices. Pure scalar math over tensor data,
/// deterministic and identical on every backend.
fn svd_host<F: Float + Copy + core::ops::AddAssign + core::ops::SubAssign>(
    a: &[F],
    m: usize,
    n: usize,
    batch: usize,
    max_sweeps: usize,
    swap: bool,
) -> Result<SvdFactors<F>, usize> {
    let mut u = vec![F::zero(); batch * m * n];
    let mut sigma = vec![F::zero(); batch * n];
    let mut vt = vec![F::zero(); batch * n * n];
    let mut d = vec![F::zero(); n];
    let mut e = vec![F::zero(); n.saturating_sub(1)];
    let mut givens = Vec::new();
    let mut order: Vec<usize> = (0..n).collect();

    for b in 0..batch {
        let ab = &a[b * m * n..(b + 1) * m * n];
        let (u1, bv, mut v1) = bidiag_host(ab, m, n);
        // Transpose the U1 factor up front: the Givens rotations act on
        // column pairs (k, k+1), which are stride-n in row-major storage but
        // contiguous rows in the transposed layout, so the hot loops below
        // stream memory instead of hopping columns. V1 already comes out of
        // bidiag_host in transposed layout.
        let mut u1t = vec![F::zero(); m * n];
        for i in 0..m {
            for j in 0..n {
                u1t[j * m + i] = u1[i * n + j];
            }
        }
        for i in 0..n {
            d[i] = bv[i * n + i];
        }
        for i in 0..n.saturating_sub(1) {
            e[i] = bv[i * n + i + 1];
        }
        givens.clear();
        if dbdsqr(&mut d, &mut e, &mut givens, max_sweeps).is_none() {
            return Err(b);
        }

        // Apply both the left (U) and right (Vt) rotations in one pass over
        // the givens list: each entry touches one row pair of each factor, so
        // a single loop keeps the rotation data in cache for both matrices.
        for &(k, cl, sl, cr, sr) in &givens {
            let (k_u, k1_u) = (k * m, (k + 1) * m);
            let (k_v, k1_v) = (k * n, (k + 1) * n);
            for i in 0..m {
                let (a0, b0) = (u1t[k_u + i], u1t[k1_u + i]);
                u1t[k_u + i] = cl * a0 + sl * b0;
                u1t[k1_u + i] = -sl * a0 + cl * b0;
            }
            for i in 0..n {
                let (a0, b0) = (v1[k_v + i], v1[k1_v + i]);
                v1[k_v + i] = cr * a0 + sr * b0;
                v1[k1_v + i] = -sr * a0 + cr * b0;
            }
        }
        svd_postprocess(
            &u1t,
            &v1,
            &d,
            &mut u[b * m * n..(b + 1) * m * n],
            &mut sigma[b * n..(b + 1) * n],
            &mut vt[b * n * n..(b + 1) * n * n],
            &mut order,
        );
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
        Ok((uf, sigma, vf))
    } else {
        Ok((u, sigma, vt))
    }
}

/// Sort the singular values descending, permute the factors and absorb the
/// diagonal signs into U.
///
/// Inputs are the transposed factor buffers (columns of U1 as rows of
/// `u1t`, rows of V1 as rows of `v1t`) and the converged diagonal `d` from
/// dbdsqr. The final row-major factors are written directly into `u`, `sigma`,
/// and `vt`; the wide-input swap is applied by the caller.
fn svd_postprocess<F: Float + Copy>(
    u1t: &[F],
    v1t: &[F],
    d: &[F],
    u: &mut [F],
    sigma: &mut [F],
    vt: &mut [F],
    order: &mut [usize],
) {
    let n = d.len();
    if n == 0 {
        return;
    }
    let m = u.len() / n;

    order.sort_unstable_by(|&i, &j| {
        d[j].abs()
            .partial_cmp(&d[i].abs())
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    // In the transposed layout, permuting U columns and Vt rows means copying
    // rows from u1t and v1t. Absorb the diagonal signs into U while copying.
    for (dest, &src) in order.iter().enumerate() {
        let flip = d[src] < F::zero();
        for i in 0..m {
            u[i * n + dest] = if flip {
                -u1t[src * m + i]
            } else {
                u1t[src * m + i]
            };
        }
        for i in 0..n {
            vt[dest * n + i] = v1t[src * n + i];
        }
        sigma[dest] = d[src].abs();
    }
}

/// Scaled norm sqrt(sum x^2) over `len` entries produced by `x`, computed
/// through a max-rescale so extreme inputs (|x| up to f32::MAX, down to
/// subnormals) can neither overflow nor underflow, like the LAPACK dlarfg
/// norm.
#[inline]
fn scaled_norm<F: Float + Copy + core::ops::AddAssign>(
    mut x: impl FnMut(usize) -> F,
    len: usize,
) -> F {
    let mut scale = F::zero();
    for k in 0..len {
        scale = scale.max(x(k).abs());
    }
    if scale == F::zero() {
        return F::zero();
    }
    let mut s = F::zero();
    for k in 0..len {
        let t = x(k) / scale;
        s += t * t;
    }
    scale * s.sqrt()
}

/// Pivot of a Householder reflection that maps the vector whose leading
/// entry is `x0` and whose norm is `norm` onto `-sign(x0) * norm * e0`
/// (LAPACK dlarfg convention): returns `(u0, tau)` where `u0` is the pivot
/// denominator and `tau` scales the rank-1 update. `tau == 0` encodes an
/// already-annihilated vector.
fn house_pivot<F: Float>(x0: F, norm: F) -> (F, F) {
    // sign = -(sign(x0)), with zero mapping to -1 (mask_fill in the tensor version).
    let sign = if x0 >= F::zero() { -F::one() } else { F::one() };
    let u0 = x0 - norm * sign;
    let tau = if norm == F::zero() {
        F::zero()
    } else {
        -u0 / (norm * sign)
    };
    (u0, tau)
}

/// Golub-Kahan bidiagonalization on the host: `A = U1 B V1^T` with `B` upper
/// bidiagonal (row-major `[m, n]`), using Householder reflections on
/// shrinking submatrices, mirroring the tensor-op version operation for
/// operation.
fn bidiag_host<F: Float + Copy + core::ops::AddAssign + core::ops::SubAssign>(
    a: &[F],
    m: usize,
    n: usize,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    // V1 kept in transposed layout (v1t[j*n+i] = V1[i,j]): the reflection
    // sweeps below then run over contiguous rows, and svd_host consumes
    // the transposed factor directly (no extra transposition pass).
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
    let mut vw = vec![F::zero(); n];

    for i in 0..n {
        // Left reflection: annihilate the subdiagonal of column i.
        let norm = scaled_norm(|k| a[(i + k) * n + i], m - i);
        let (u0, tau) = house_pivot(a[i * n + i], norm);
        if tau != F::zero() {
            w[i] = F::one();
            for k in (i + 1)..m {
                w[k] = a[k * n + i] / u0;
            }
            // wta[j] = w^T a[:, j], then a_new = a - tau w wta. Row-major
            // sweeps: a[k, :] is contiguous, so both passes stay sequential.
            for v in wta[i..n].iter_mut() {
                *v = F::zero();
            }
            for k in i..m {
                let wk = w[k];
                for j in i..n {
                    wta[j] += wk * a[k * n + j];
                }
            }
            for k in i..m {
                let wk = w[k] * tau;
                for j in i..n {
                    a[k * n + j] -= wk * wta[j];
                }
            }
            for (k, &wk) in w.iter().enumerate().take(m) {
                ws[i * m + k] = wk;
            }
            taus[i] = tau;
        }

        // Right reflection: annihilate row i right of the superdiagonal.
        if i + 2 < n {
            let norm = scaled_norm(|j| a[i * n + i + 1 + j], n - i - 1);
            let (u0, tau) = house_pivot(a[i * n + i + 1], norm);
            if tau != F::zero() {
                w[i + 1] = F::one();
                for j in (i + 2)..n {
                    w[j] = a[i * n + j] / u0;
                }
                // aw[k] = a[k, :] w, then a_new = a - tau aw w^T. Contiguous
                // row sweeps again: a[k, j] advances sequentially in both.
                for k in i..m {
                    aw[k] = F::zero();
                    for j in (i + 1)..n {
                        aw[k] += a[k * n + j] * w[j];
                    }
                }
                for k in i..m {
                    let awk = aw[k] * tau;
                    for j in (i + 1)..n {
                        a[k * n + j] -= awk * w[j];
                    }
                }
                // V1 = V1 (I - tau w w^T) on the transposed factor: v1t[j, i2]
                // -= tau w[j] vw[i2] with vw[i2] = sum_j v1t[j, i2] w[j].
                // Both sweeps are outer-j/inner-i2, so they stream contiguous
                // rows and vectorize (accumulators live in the vw array, no
                // scalar reduction).
                for (j, &wj) in w[i + 1..n].iter().enumerate() {
                    let row = (i + 1 + j) * n;
                    for i2 in 0..n {
                        vw[i2] += v1[row + i2] * wj;
                    }
                }
                for (j, &wj) in w[i + 1..n].iter().enumerate() {
                    let row = (i + 1 + j) * n;
                    for i2 in 0..n {
                        v1[row + i2] -= tau * wj * vw[i2];
                    }
                }
                vw.fill(F::zero());
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
                    uw[j] += wk * u1[k * n + j];
                }
            }
            let t = tau;
            for k in i..m {
                let wk = ws[i * m + k] * t;
                for j in 0..n {
                    u1[k * n + j] -= wk * uw[j];
                }
            }
        }
    }
    (u1, a, v1)
}

/// LAPACK dbdsqr-style shifted QR iteration on an upper bidiagonal matrix.
/// Mutates `d` to the signed singular values and logs the Givens rotations in
/// application order so the caller can rebuild the singular vectors. Returns
/// `None` when an active block remains after the sweep budget is exhausted.
fn dbdsqr<F: Float + Copy>(
    d: &mut [F],
    e: &mut [F],
    givens: &mut Vec<GivensRotation<F>>,
    max_sweeps: usize,
) -> Option<()> {
    let n = d.len();
    let eps = F::epsilon();
    let tol = eps * F::from(10.0).unwrap();
    let mut smax = F::zero();
    for &x in d.iter().chain(e.iter()) {
        smax = smax.max(x.abs());
    }
    if smax == F::zero() {
        return Some(());
    }
    let mut m = n;
    let mut iters = 0;
    let max_iters = max_sweeps.saturating_mul(n);
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
        if iters >= max_iters {
            return None;
        }
        // LAPACK dbdsqr uses the smaller singular value of the trailing 2x2
        // block as its shift. Compute it in f64 so f32 rounding cannot stall
        // deflation near the tolerance boundary.
        let shift = F::from(dlas2_smin(
            d[m - 2].to_f64().unwrap(),
            e[m - 2].to_f64().unwrap(),
            d[m - 1].to_f64().unwrap(),
        ))
        .unwrap();
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
    }
    Some(())
}

/// A single Givens rotation produced by [`dbdsqr`]: the pivot column
/// pair and the four rotation coefficients (cosl, sinl, cosr, sinr).
type GivensRotation<F> = (usize, F, F, F, F);

/// Smaller singular value of the 2x2 block [[d1, e1], [0, d2]], the LAPACK
/// dbdsqr shift. SSMIN = |d1 d2| / SSMAX is exact (the roots of the trailing
/// 2x2 of B^T B multiply to d1^2 d2^2), and dividing after scaling keeps both
/// the product and the quotient free of overflow/underflow.
fn dlas2_smin(d1: f64, e1: f64, d2: f64) -> f64 {
    let smax = dlas2_smax(d1, e1, d2);
    if smax == 0.0 {
        0.0
    } else {
        (d1 * (d2 / smax)).abs()
    }
}

/// Largest singular value of the 2x2 block [[d1, e1], [0, d2]]. Scaled like
/// LAPACK dlas2: no overflow or underflow on the intermediate squares.
fn dlas2_smax(d1: f64, e1: f64, d2: f64) -> f64 {
    let t = d1 * d1 + d2 * d2 + e1 * e1;
    let disc = t * t - 4.0 * d1 * d1 * d2 * d2;
    // Fast path only when the discriminant is computable: for f64 entries in
    // ~[7.7e76, 1.3e154], t is finite but t*t overflows, and max(NaN, 0) = 0
    // would silently return a wrong (low or inf) result. disc < 0 is normal
    // (rounding) and clamps to 0; only non-finite values need the scaled path.
    if t.is_finite() && disc.is_finite() {
        ((t + disc.max(0.0).sqrt()) / 2.0).max(0.0).sqrt()
    } else {
        let scale = d1.abs().max(d2.abs()).max(e1.abs());
        if scale == 0.0 {
            return 0.0;
        }
        let (a, b, c) = (d1 / scale, d2 / scale, e1 / scale);
        let t = a * a + b * b + c * c;
        let disc = (t * t - 4.0 * a * a * b * b).max(0.0);
        scale * ((t + disc.sqrt()) / 2.0).sqrt()
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

    /// Defensive numeric paths of the LAPACK-style helpers, driven directly:
    /// all-zero and overflow-scale 2x2 shifts, underflowing Givens
    /// generation, and the sweep budget bail-out.
    #[test]
    fn test_svd_host_numeric_guards() {
        let run_dbdsqr = |mut d: Vec<f64>, mut e: Vec<f64>, sweeps| {
            dbdsqr(&mut d, &mut e, &mut Vec::new(), sweeps)
        };

        // dlas2_smax: all-zero block and an f64 block whose squares would
        // overflow (t ~ 3e400) force the scaled path.
        assert_eq!(dlas2_smax(0.0f64, 0.0, 0.0), 0.0);
        assert_eq!(dlas2_smin(0.0f64, 0.0, 0.0), 0.0);
        let smax = dlas2_smax(1e200f64, 1e200, 1e200);
        assert!(smax.is_finite() && smax > 1e199 && smax < 2e200, "{smax}");
        // dlartg: zero vector and an f32 pair whose squares underflow to 0.
        let (c, s, r) = dlartg(0.0f32, 0.0);
        assert_eq!((c, s, r), (1.0, 0.0, 0.0));
        let (c, s, r) = dlartg(1e-30f32, 1e-30);
        assert!(r.is_finite() && r > 0.0, "{r}");
        assert!((c * c + s * s - 1.0).abs() < 1e-5, "({c}, {s})");
        // An active block cannot be returned as a completed SVD when no QR
        // sweeps are available.
        assert!(run_dbdsqr(vec![3.0, 1.0], vec![0.5], 0).is_none());

        // The host batch boundary reports the first matrix that did not
        // converge. Batch 0 is already diagonal; batch 1 needs a QR sweep.
        let a = [1.0f64, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0];
        assert!(matches!(svd_host(&a, 2, 2, 2, 0, false), Err(1)));

        // A positive budget can also be exhausted; partial factors must not
        // be returned as a completed decomposition.
        assert!(
            run_dbdsqr(vec![1.0; 6], vec![1.0; 5], 1).is_none(),
            "one sweep per singular value should be insufficient"
        );
    }

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
        let (u, s, vt) = svd_host::<f64>(&a, 4, 3, 1, 30, false).unwrap();
        // Reference from numpy/LAPACK gesdd (svd_host already sorts descending).
        assert!((s[0] - 25.46240743603639).abs() < 1e-12, "s1 {}", s[0]);
        assert!((s[1] - 1.290661675761233).abs() < 1e-12, "s2 {}", s[1]);
        assert!(recon_err::<f64>(&u, &s, &vt, &a, 4, 3) < 1e-12);
        // 1x1 and rank-1 edge cases
        let (u, s, vt) = svd_host::<f64>(&[-3.0], 1, 1, 1, 30, false).unwrap();
        assert!((s[0] - 3.0).abs() < 1e-15);
        assert!(recon_err::<f64>(&u, &s, &vt, &[-3.0], 1, 1) < 1e-15);
        // rank-1 3x2 (m >= n as svd_host requires; wide inputs are transposed
        // in svd() itself)
        let a = [1.0f64, 2.0, 2.0, 4.0, 3.0, 6.0];
        let (u, s, vt) = svd_host::<f64>(&a, 3, 2, 1, 30, false).unwrap();
        assert!(recon_err::<f64>(&u, &s, &vt, &a, 3, 2) < 1e-12);
        // batched 4x3
        let a = [
            1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.5, 2.5,
        ];
        let (u, s, vt) = svd_host::<f64>(&a, 4, 3, 2, 30, false).unwrap();
        assert!(recon_err::<f64>(&u[..12], &s[..3], &vt[..9], &a[..12], 4, 3) < 1e-12);
        assert!(recon_err::<f64>(&u[12..], &s[3..], &vt[9..], &a[12..], 4, 3) < 1e-12);
        // extreme scales stay finite
        let a = [1e200f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (u, s, vt) = svd_host::<f64>(&a, 3, 3, 1, 30, false).unwrap();
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
            let (u, s, vt) = svd_host::<f32>(&a, 3, 3, 1, 30, false).unwrap();
            for x in s.iter() {
                assert!(x.is_finite(), "sigma not finite: {:?}", s);
            }
            let err = recon_err::<f32>(&u, &s, &vt, &a, 3, 3);
            assert!(err.is_finite(), "recon not finite {err}");
            assert!(err <= a[0].abs().max(1.0) * 1e-4, "recon err {err}");
        }

        let a = [1e38f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (_, s, _) = svd_host::<f32>(&a, 3, 3, 1, 30, false).unwrap();
        assert_eq!(s, [1e38, 0.0, 0.0]);
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
        let (u, s, vt) = svd_host::<f64>(&a, m, n, 1, 30, false).unwrap();
        assert!(recon_err::<f64>(&u, &s, &vt, &a, m, n) < 1e-9);

        // same matrix in f32
        let af: Vec<f32> = a.iter().map(|x| *x as f32).collect();
        let (u, s, vt) = svd_host::<f32>(&af, m, n, 1, 30, false).unwrap();
        assert!(recon_err::<f32>(&u, &s, &vt, &af, m, n) < 1e-3);
    }

    #[test]
    fn test_svd_host_2x2_direct() {
        // B = [[2,1],[0,1]].
        let a = [2.0f64, 1.0, 0.0, 1.0];
        let (u, s, vt) = svd_host::<f64>(&a, 2, 2, 1, 30, false).unwrap();
        assert!(recon_err::<f64>(&u, &s, &vt, &a, 2, 2) < 1e-12);

        // Negative determinant: the right factor must still reconstruct A.
        // (A regression test: a previous "handedness fix" negated Vt row 1
        // without U column 1, breaking every det < 0 input.)
        for a in [
            [1.0f64, 2.0, 3.0, 4.0],
            [0.0f64, 1.0, 1.0, 0.0],
            [-3.0f64, 1.0, 2.0, -1.0],
        ] {
            let (u, s, vt) = svd_host::<f64>(&a, 2, 2, 1, 30, false).unwrap();
            assert!(recon_err::<f64>(&u, &s, &vt, &a, 2, 2) < 1e-12);
        }

        // Large finite values must not overflow intermediate calculations.
        let a = [1e200f64, 0.0, 0.0, 1e200];
        let (u, s, vt) = svd_host::<f64>(&a, 2, 2, 1, 30, false).unwrap();
        assert!(u.iter().chain(&s).chain(&vt).all(|x| x.is_finite()));
        assert!(recon_err::<f64>(&u, &s, &vt, &a, 2, 2) < 1e187);

        // SVD must preserve representable non-zero singular values rather
        // than applying an implicit rank threshold.
        let a = [1.0f32, 0.0, 0.0, 1e-7];
        let (_, s, _) = svd_host::<f32>(&a, 2, 2, 1, 30, false).unwrap();
        assert!((s[1] - 1e-7).abs() < 1e-12, "singular values {s:?}");
    }

    #[test]
    fn test_svd_host_2x2_rank_deficient_orthonormal() {
        // Rank-deficient inputs may produce a tiny numerical residual, but
        // the factors must remain orthonormal and reconstruct the input.
        for a in [
            [1.0f64, 2.0, 2.0, 4.0],
            [0.0f64, 1.0, 0.0, 0.0],
            [0.0f64, 0.0, 1.0, 0.0],
            [1.0f64, 1.0, 2.0, 2.0],
        ] {
            let (u, s, vt) = svd_host::<f64>(&a, 2, 2, 1, 30, false).unwrap();
            assert!(s[1] <= s[0] * 1e-14, "singular values for {a:?}: {s:?}");
            // Vt rows orthonormal: Vt Vt^T = I.
            let ortho = (0..2)
                .map(|i| {
                    (0..2)
                        .map(|j| {
                            let dot = (0..2).map(|k| vt[i * 2 + k] * vt[j * 2 + k]).sum::<f64>();
                            (dot - if i == j { 1.0 } else { 0.0 }).abs()
                        })
                        .fold(0.0f64, f64::max)
                })
                .fold(0.0f64, f64::max);
            assert!(ortho < 1e-12, "Vt orthonormal for {a:?}, err {ortho}");
            assert!(recon_err::<f64>(&u, &s, &vt, &a, 2, 2) < 1e-12);
        }
    }

    #[test]
    fn test_svd_host_zero_m1_orthonormal() {
        // Zero m x 1 matrix: U must stay orthonormal (unit basis), not an
        // all-ones column of norm sqrt(m).
        let (m, n) = (5usize, 1usize);
        let a = [0.0f64; 5];
        let (u, s, vt) = svd_host::<f64>(&a, m, n, 1, 30, false).unwrap();
        assert_eq!(s[0], 0.0);
        let norm: f64 = u.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 1e-15, "U column norm {norm}");
        assert_eq!(vt[0], 1.0);
        // Non-zero column keeps the normalized-column form.
        let a = [0.0f64, 3.0, 0.0, 4.0, 0.0];
        let (u, s, vt) = svd_host::<f64>(&a, m, n, 1, 30, false).unwrap();
        assert!((s[0] - 5.0).abs() < 1e-15);
        let mut err = 0.0f64;
        for i in 0..m {
            err = err.max((a[i] - u[i * n] * s[0] * vt[0]).abs());
        }
        assert!(err < 1e-15, "m1 recon {err}");
    }
}
