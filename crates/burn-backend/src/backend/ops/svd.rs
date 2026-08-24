//! Host SVD pipeline: Golub-Kahan bidiagonalization + LAPACK-style `dbdsqr`
//! + Givens factor assembly, over plain slices.
//!
//! This is the reference implementation backing the default
//! [`FloatTensorOps::float_svd`](super::tensor::FloatTensorOps#method.float_svd):
//! plain-slice math over tensor data with runtime-detected AVX2 kernels for
//! the hot loops (bit-identical to the portable fallback), deterministic and
//! identical on every backend. Backends may override the trait method with a
//! native SVD (tch) or a fused GPU kernel (cubecl); this module stays the
//! correctness reference and the no-kernel fallback.
use alloc::vec;
use alloc::vec::Vec;
use burn_std::{DType, TensorData};
use num_traits::float::Float;

/// Whether the process may run the AVX2 kernel bodies for the hot loops.
///
/// Every kernel below has a portable fallback; both bodies perform the same
/// per-element operations in the same order (independent elements, or eight
/// independent accumulator chains reduced left-to-right), so the output is
/// bit-identical no matter which one runs. Runtime-detected once.
///
/// Test-only override: 1 forces AVX2, 2 forces the fallback, any other value
/// follows runtime detection. Used to prove both paths are bit-identical.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn simd_fast() -> bool {
    #[cfg(test)]
    match SIMD_TEST_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    static AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVX2.get_or_init(|| is_x86_feature_detected!("avx2"))
}

#[cfg(all(test, target_arch = "x86_64", feature = "std"))]
static SIMD_TEST_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// `y[i] += t * x[i]` over the full slices.
#[inline(always)]
fn slice_axpy<F: Float + Copy>(y: &mut [F], x: &[F], t: F) {
    debug_assert_eq!(y.len(), x.len());
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if simd_fast() {
        // Safety: only `f32`/`f64` element types exist here (asserted); `y`
        // and `x` point to `len` live, non-overlapping elements of distinct
        // buffers at every call site.
        unsafe {
            if core::mem::size_of::<F>() == 4 {
                simd::axpy(
                    y.as_mut_ptr().cast::<f32>(),
                    x.as_ptr().cast::<f32>(),
                    t.to_f32().unwrap(),
                    y.len(),
                );
            } else {
                debug_assert_eq!(core::mem::size_of::<F>(), 8);
                simd::axpy_f64(
                    y.as_mut_ptr().cast::<f64>(),
                    x.as_ptr().cast::<f64>(),
                    t.to_f64().unwrap(),
                    y.len(),
                );
            }
        }
        return;
    }
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = *yi + t * xi;
    }
}

/// Dot product `sum a[i] * w[i]` over `lo..hi`, accumulated in independent
/// chains reduced in fixed left-to-right order so LLVM can vectorize without
/// fast-math. The chain count matches the AVX2 body's register width for the
/// element type (8 for f32, 4 for f64).
#[inline(always)]
fn slice_dot<F: Float + Copy + core::ops::AddAssign>(a: &[F], w: &[F], lo: usize, hi: usize) -> F {
    debug_assert!(lo <= hi && hi <= a.len() && hi <= w.len());
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if simd_fast() {
        // Safety: `lo..hi` is in bounds for both buffers; reads only.
        unsafe {
            if core::mem::size_of::<F>() == 4 {
                let r = simd::dot(a.as_ptr().cast::<f32>(), w.as_ptr().cast::<f32>(), lo, hi);
                return F::from(r).unwrap();
            } else {
                debug_assert_eq!(core::mem::size_of::<F>(), 8);
                let r = simd::dot_f64(a.as_ptr().cast::<f64>(), w.as_ptr().cast::<f64>(), lo, hi);
                return F::from(r).unwrap();
            }
        }
    }
    // Chain count matches the AVX2 body's register width per element type,
    // so both accumulation orders are identical.
    let lanes = if core::mem::size_of::<F>() == 4 { 8 } else { 4 };
    let mut s = [F::zero(); 8];
    let mut j = lo;
    while j + lanes <= hi {
        for t in 0..lanes {
            s[t] += a[j + t] * w[j + t];
        }
        j += lanes;
    }
    let mut r = s[0];
    for &chain in s.iter().take(lanes).skip(1) {
        r += chain;
    }
    while j < hi {
        r += a[j] * w[j];
        j += 1;
    }
    r
}

/// Scaled sum of squares `sum (x[i] / scale)^2` over `lo..hi`, same
/// structure as [`slice_dot`].
#[inline(always)]
fn slice_sumsq_scaled<F: Float + Copy + core::ops::AddAssign>(
    x: &[F],
    scale: F,
    lo: usize,
    hi: usize,
) -> F {
    debug_assert!(lo <= hi && hi <= x.len());
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if simd_fast() {
        // Safety: `lo..hi` is in bounds; reads only.
        unsafe {
            if core::mem::size_of::<F>() == 4 {
                let r =
                    simd::sumsq_scaled(x.as_ptr().cast::<f32>(), scale.to_f32().unwrap(), lo, hi);
                return F::from(r).unwrap();
            } else {
                debug_assert_eq!(core::mem::size_of::<F>(), 8);
                let r = simd::sumsq_scaled_f64(
                    x.as_ptr().cast::<f64>(),
                    scale.to_f64().unwrap(),
                    lo,
                    hi,
                );
                return F::from(r).unwrap();
            }
        }
    }
    let lanes = if core::mem::size_of::<F>() == 4 { 8 } else { 4 };
    let mut s = [F::zero(); 8];
    let mut j = lo;
    while j + lanes <= hi {
        for t in 0..lanes {
            let d = x[j + t] / scale;
            s[t] += d * d;
        }
        j += lanes;
    }
    let mut r = s[0];
    for &chain in s.iter().take(lanes).skip(1) {
        r += chain;
    }
    while j < hi {
        let d = x[j] / scale;
        r += d * d;
        j += 1;
    }
    r
}

/// Apply one Givens rotation to two contiguous rows of `buf`:
/// `(a, b) <- (c*a + s*b, -s*a + c*b)` element-wise over `len` entries
/// starting at offsets `i0` and `i1`.
///
/// The two rows must not overlap; every call site rotates distinct adjacent
/// rows of a single buffer.
#[inline(always)]
fn rotate_rows<F: Float + Copy>(buf: &mut [F], i0: usize, i1: usize, c: F, s: F, len: usize) {
    debug_assert!(i0.checked_add(len).is_some_and(|e| e <= buf.len()));
    debug_assert!(i1.checked_add(len).is_some_and(|e| e <= buf.len()));
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    if simd_fast() {
        // Safety: both ranges are in bounds (asserted above); the call sites
        // rotate disjoint adjacent rows of one buffer.
        unsafe {
            if core::mem::size_of::<F>() == 4 {
                simd::rotate(
                    buf.as_mut_ptr().cast::<f32>(),
                    i0,
                    i1,
                    c.to_f32().unwrap(),
                    s.to_f32().unwrap(),
                    len,
                );
            } else {
                debug_assert_eq!(core::mem::size_of::<F>(), 8);
                simd::rotate_f64(
                    buf.as_mut_ptr().cast::<f64>(),
                    i0,
                    i1,
                    c.to_f64().unwrap(),
                    s.to_f64().unwrap(),
                    len,
                );
            }
        }
        return;
    }
    for t in 0..len {
        let (a0, b0) = (buf[i0 + t], buf[i1 + t]);
        buf[i0 + t] = c * a0 + s * b0;
        buf[i1 + t] = -s * a0 + c * b0;
    }
}

/// AVX2 bodies for the kernels above. Each routine performs exactly the
/// operations of its fallback (including the sequential tail and the
/// left-to-right chain reduction), so results match bit for bit. No FMA
/// contraction is used anywhere: rustc never contracts automatically and
/// the intrinsics below use separate mul/add.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
mod simd {
    use core::arch::x86_64::*;

    /// # Safety
    /// `y` and `x` must be valid for `len` elements and must not overlap.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn axpy(y: *mut f32, x: *const f32, t: f32, len: usize) {
        unsafe {
            let vt = _mm256_set1_ps(t);
            let mut j = 0;
            while j + 8 <= len {
                let xv = _mm256_loadu_ps(x.add(j));
                let yv = _mm256_loadu_ps(y.add(j));
                _mm256_storeu_ps(y.add(j), _mm256_add_ps(yv, _mm256_mul_ps(vt, xv)));
                j += 8;
            }
            while j < len {
                *y.add(j) = *y.add(j) + t * *x.add(j);
                j += 1;
            }
        }
    }

    /// # Safety
    /// `y` and `x` must be valid for `len` elements and must not overlap.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn axpy_f64(y: *mut f64, x: *const f64, t: f64, len: usize) {
        unsafe {
            let vt = _mm256_set1_pd(t);
            let mut j = 0;
            while j + 4 <= len {
                let xv = _mm256_loadu_pd(x.add(j));
                let yv = _mm256_loadu_pd(y.add(j));
                _mm256_storeu_pd(y.add(j), _mm256_add_pd(yv, _mm256_mul_pd(vt, xv)));
                j += 4;
            }
            while j < len {
                *y.add(j) = *y.add(j) + t * *x.add(j);
                j += 1;
            }
        }
    }

    /// # Safety
    /// `a` and `w` must be valid for `hi` elements.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn dot(a: *const f32, w: *const f32, lo: usize, hi: usize) -> f32 {
        unsafe {
            let mut acc = _mm256_setzero_ps();
            let mut j = lo;
            while j + 8 <= hi {
                acc = _mm256_add_ps(
                    acc,
                    _mm256_mul_ps(_mm256_loadu_ps(a.add(j)), _mm256_loadu_ps(w.add(j))),
                );
                j += 8;
            }
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
            let mut r = tmp[0];
            r += tmp[1];
            r += tmp[2];
            r += tmp[3];
            r += tmp[4];
            r += tmp[5];
            r += tmp[6];
            r += tmp[7];
            while j < hi {
                r += *a.add(j) * *w.add(j);
                j += 1;
            }
            r
        }
    }

    /// # Safety
    /// `a` and `w` must be valid for `hi` elements.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn dot_f64(a: *const f64, w: *const f64, lo: usize, hi: usize) -> f64 {
        unsafe {
            let mut acc = _mm256_setzero_pd();
            let mut j = lo;
            while j + 4 <= hi {
                acc = _mm256_add_pd(
                    acc,
                    _mm256_mul_pd(_mm256_loadu_pd(a.add(j)), _mm256_loadu_pd(w.add(j))),
                );
                j += 4;
            }
            let mut tmp = [0.0f64; 4];
            _mm256_storeu_pd(tmp.as_mut_ptr(), acc);
            let mut r = tmp[0];
            r += tmp[1];
            r += tmp[2];
            r += tmp[3];
            while j < hi {
                r += *a.add(j) * *w.add(j);
                j += 1;
            }
            r
        }
    }

    /// # Safety
    /// `x` must be valid for `hi` elements; `scale` must be non-zero.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn sumsq_scaled(x: *const f32, scale: f32, lo: usize, hi: usize) -> f32 {
        unsafe {
            let vscale = _mm256_set1_ps(scale);
            let mut acc = _mm256_setzero_ps();
            let mut j = lo;
            while j + 8 <= hi {
                let d = _mm256_div_ps(_mm256_loadu_ps(x.add(j)), vscale);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(d, d));
                j += 8;
            }
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
            let mut r = tmp[0];
            r += tmp[1];
            r += tmp[2];
            r += tmp[3];
            r += tmp[4];
            r += tmp[5];
            r += tmp[6];
            r += tmp[7];
            while j < hi {
                let d = *x.add(j) / scale;
                r += d * d;
                j += 1;
            }
            r
        }
    }

    /// # Safety
    /// `x` must be valid for `hi` elements; `scale` must be non-zero.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn sumsq_scaled_f64(x: *const f64, scale: f64, lo: usize, hi: usize) -> f64 {
        unsafe {
            let vscale = _mm256_set1_pd(scale);
            let mut acc = _mm256_setzero_pd();
            let mut j = lo;
            while j + 4 <= hi {
                let d = _mm256_div_pd(_mm256_loadu_pd(x.add(j)), vscale);
                acc = _mm256_add_pd(acc, _mm256_mul_pd(d, d));
                j += 4;
            }
            let mut tmp = [0.0f64; 4];
            _mm256_storeu_pd(tmp.as_mut_ptr(), acc);
            let mut r = tmp[0];
            r += tmp[1];
            r += tmp[2];
            r += tmp[3];
            while j < hi {
                let d = *x.add(j) / scale;
                r += d * d;
                j += 1;
            }
            r
        }
    }

    /// # Safety
    /// Ranges `[i0, i0+len)` and `[i1, i1+len)` must lie inside the same
    /// allocation and must not overlap.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn rotate(buf: *mut f32, i0: usize, i1: usize, c: f32, s: f32, len: usize) {
        unsafe {
            let vc = _mm256_set1_ps(c);
            let vs = _mm256_set1_ps(s);
            let vns = _mm256_set1_ps(-s);
            let mut j = 0;
            while j + 8 <= len {
                let pa = buf.add(i0 + j);
                let pb = buf.add(i1 + j);
                let va = _mm256_loadu_ps(pa);
                let vb = _mm256_loadu_ps(pb);
                _mm256_storeu_ps(
                    pa,
                    _mm256_add_ps(_mm256_mul_ps(vc, va), _mm256_mul_ps(vs, vb)),
                );
                _mm256_storeu_ps(
                    pb,
                    _mm256_add_ps(_mm256_mul_ps(vns, va), _mm256_mul_ps(vc, vb)),
                );
                j += 8;
            }
            while j < len {
                let (a0, b0) = (*buf.add(i0 + j), *buf.add(i1 + j));
                *buf.add(i0 + j) = c * a0 + s * b0;
                *buf.add(i1 + j) = -s * a0 + c * b0;
                j += 1;
            }
        }
    }

    /// # Safety
    /// Ranges `[i0, i0+len)` and `[i1, i1+len)` must lie inside the same
    /// allocation and must not overlap.
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn rotate_f64(
        buf: *mut f64,
        i0: usize,
        i1: usize,
        c: f64,
        s: f64,
        len: usize,
    ) {
        unsafe {
            let vc = _mm256_set1_pd(c);
            let vs = _mm256_set1_pd(s);
            let vns = _mm256_set1_pd(-s);
            let mut j = 0;
            while j + 4 <= len {
                let pa = buf.add(i0 + j);
                let pb = buf.add(i1 + j);
                let va = _mm256_loadu_pd(pa);
                let vb = _mm256_loadu_pd(pb);
                _mm256_storeu_pd(
                    pa,
                    _mm256_add_pd(_mm256_mul_pd(vc, va), _mm256_mul_pd(vs, vb)),
                );
                _mm256_storeu_pd(
                    pb,
                    _mm256_add_pd(_mm256_mul_pd(vns, va), _mm256_mul_pd(vc, vb)),
                );
                j += 4;
            }
            while j < len {
                let (a0, b0) = (*buf.add(i0 + j), *buf.add(i1 + j));
                *buf.add(i0 + j) = c * a0 + s * b0;
                *buf.add(i1 + j) = -s * a0 + c * b0;
                j += 1;
            }
        }
    }
}

/// Run the full host pipeline over tensor data and return the three factors
/// as data. Layout and dims follow the `swap` convention of
/// `crates/burn-tensor`'s `linalg::svd` (see `svd_host`).
///
/// Consumes the data so the input can be converted to a plain vector without
/// copying (the backend transfers are already owned by the caller).
pub(crate) fn svd_host_data(
    data: TensorData,
    sweeps: usize,
    swap: bool,
) -> (TensorData, TensorData, TensorData) {
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
        let a = data.into_vec::<f64>().unwrap();
        let (u, s, vt) = svd_host::<f64>(&a, m, n, batch, sweeps, swap);
        (
            TensorData::new(u, du),
            TensorData::new(s, ds),
            TensorData::new(vt, dv),
        )
    } else {
        let a = data.into_vec::<f32>().unwrap();
        let (u, s, vt) = svd_host::<f32>(&a, m, n, batch, sweeps, swap);
        (
            TensorData::new(u, du),
            TensorData::new(s, ds),
            TensorData::new(vt, dv),
        )
    }
}

/// Host pipeline, per batch element. Pure scalar math over tensor data,
/// deterministic and identical on every backend.
///
/// With the `std` feature the batch is split across threads: the matrices in
/// a batch are independent, so each thread works on its own slice and the
/// results are concatenated in batch order (deterministic, same output).
/// No-std builds fall back to the single-threaded loop.
fn svd_host<F: Float + Copy + Send + Sync + core::ops::AddAssign>(
    a: &[F],
    m: usize,
    n: usize,
    batch: usize,
    max_sweeps: usize,
    swap: bool,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    #[cfg(feature = "std")]
    if batch > 1 {
        // Factor layouts depend on the orientation: without swap U is [m, n]
        // and Vt is [n, n]; with swap (wide input) U is [n, n] and Vt is [m, n].
        // Only the parallel path needs the preallocated buffers (it splits
        // them up front); the serial path returns svd_host_seq's fresh Vecs
        // directly.
        let (u_len, vt_len) = if swap { (n * n, m * n) } else { (m * n, n * n) };
        // Thread spawn+join costs ~10-40us; a single 4x4 element is ~3us, so
        // parallelizing tiny batches makes them slower, not faster. Only
        // spawn when the total work is worth it (~64x64 serial is ~120us).
        // Element count is a solid work proxy across shapes. The closed forms
        // (n == 1, 2x2) are allocation-bound: spawning for e.g. 4096
        // 1x1 elements is a net loss, so they stay serial.
        if batch * m * n >= 4096 && !(n == 1 || (m == 2 && n == 2)) {
            // available_parallelism is a syscall (~40us); skip it for single
            // matrices where the threaded path is dead anyway.
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(batch)
                .max(1);
            if threads > 1 {
                let mut u = vec![F::zero(); batch * u_len];
                let mut sigma = vec![F::zero(); batch * n];
                let mut vt = vec![F::zero(); batch * vt_len];
                // Round-robin over single batch elements: thread t handles
                // elements t, t + threads, ... This uses all cores even when
                // batch is not a multiple of threads (static contiguous
                // chunks would leave cores idle), and a slow (pathological)
                // element only delays its own thread, never a whole chunk.
                // Slices are split up front so every spawned closure holds
                // only its own &mut borrows; output order is still batch
                // order (each element writes its own disjoint slices).
                let mut u_lists: Vec<Vec<&mut [F]>> = (0..threads).map(|_| Vec::new()).collect();
                let mut s_lists: Vec<Vec<&mut [F]>> = (0..threads).map(|_| Vec::new()).collect();
                let mut v_lists: Vec<Vec<&mut [F]>> = (0..threads).map(|_| Vec::new()).collect();
                {
                    let mut rest_u = u.as_mut_slice();
                    let mut rest_s = sigma.as_mut_slice();
                    let mut rest_v = vt.as_mut_slice();
                    for b in 0..batch {
                        let (u_part, r) = rest_u.split_at_mut(u_len);
                        let (s_part, r2) = rest_s.split_at_mut(n);
                        let (v_part, r3) = rest_v.split_at_mut(vt_len);
                        u_lists[b % threads].push(u_part);
                        s_lists[b % threads].push(s_part);
                        v_lists[b % threads].push(v_part);
                        rest_u = r;
                        rest_s = r2;
                        rest_v = r3;
                    }
                }
                std::thread::scope(|scope| {
                    for (t, (u_list, (s_list, v_list))) in u_lists
                        .into_iter()
                        .zip(s_lists.into_iter().zip(v_lists))
                        .enumerate()
                    {
                        scope.spawn(move || {
                            for (k, (u_part, (s_part, v_part))) in u_list
                                .into_iter()
                                .zip(s_list.into_iter().zip(v_list))
                                .enumerate()
                            {
                                let b = t + k * threads;
                                let a_slice = &a[b * m * n..(b + 1) * m * n];
                                let (tu, ts, tv) =
                                    svd_host_seq(a_slice, m, n, 1, max_sweeps, swap, false);
                                u_part.copy_from_slice(&tu);
                                s_part.copy_from_slice(&ts);
                                v_part.copy_from_slice(&tv);
                            }
                        });
                    }
                });
                return (u, sigma, vt);
            }
        }
    }

    svd_host_seq(a, m, n, batch, max_sweeps, swap, true)
}

/// Per-batch pipeline (shared by the parallel wrapper and the serial path).
///
/// With `par_givens` and the `std` feature, single matrices large enough
/// split the U and Vt rotation applications across two scoped threads; the
/// batch wrapper runs this with `false` so worker threads never nest spawns.
fn svd_host_seq<F: Float + Copy + Send + Sync + core::ops::AddAssign>(
    a: &[F],
    m: usize,
    n: usize,
    batch: usize,
    max_sweeps: usize,
    swap: bool,
    par_givens: bool,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    #[cfg(not(feature = "std"))]
    let _ = par_givens;
    let mut u = vec![F::zero(); batch * m * n];
    let mut sigma = vec![F::zero(); batch * n];
    let mut vt = vec![F::zero(); batch * n * n];
    let mut d = vec![F::zero(); n];
    let mut e = vec![F::zero(); n.saturating_sub(1)];

    for b in 0..batch {
        let ab = &a[b * m * n..(b + 1) * m * n];
        // Closed forms for the smallest sizes: exact and much cheaper than
        // the bidiagonalization + QR pipeline.
        if n == 1 {
            // A is [m, 1] (wide inputs were transposed): the single singular
            // value is the column norm and U is the normalized column.
            // Scaled norm (same pattern as bidiag_host): no overflow on
            // entries up to f32::MAX.
            let scale = ab.iter().fold(F::zero(), |m, &t| m.max(t.abs()));
            let sv = if scale == F::zero() {
                F::zero()
            } else {
                scale * slice_sumsq_scaled(ab, scale, 0, m).sqrt()
            };
            sigma[b] = sv;
            for i in 0..m {
                u[b * m * n + i * n] = if sv > F::zero() {
                    ab[i] / sv
                } else if i == 0 {
                    // Zero column: keep U orthonormal with a unit basis vector
                    // instead of an all-ones column of norm sqrt(m).
                    F::one()
                } else {
                    F::zero()
                };
            }
            vt[b] = F::one();
            continue;
        }
        if n == 2 && m == 2 {
            let (s0, s1, uu, vv) = svd2x2(ab);
            // Mask numerical zeros like the general path: values at or below
            // 10 * eps * sigma_max are returned as 0 (documented contract).
            let mask = s0 * (F::epsilon() * F::from(10.0).unwrap());
            sigma[b * 2] = s0;
            sigma[b * 2 + 1] = if s1 <= mask { F::zero() } else { s1 };
            u[b * 4..b * 4 + 4].copy_from_slice(&uu);
            vt[b * 4..b * 4 + 4].copy_from_slice(&vv);
            continue;
        }
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
        let (sigma_b, d_final, givens_b) = dbdsqr_host(&d, &e, max_sweeps);

        // Apply the left (U) and right (Vt) rotations. On `std`, a single
        // large matrix splits the two factor applications across two scoped
        // threads: the buffers are independent and each thread applies the
        // same rotation list in the same order, so the result is identical to
        // the serial path. Batch workers pass `par_givens = false` so they
        // never nest spawns.
        #[cfg(not(feature = "std"))]
        for &(k, cl, sl, cr, sr) in &givens_b {
            rotate_rows(&mut u1t, k * m, (k + 1) * m, cl, sl, m);
            rotate_rows(&mut v1, k * n, (k + 1) * n, cr, sr, n);
        }
        #[cfg(feature = "std")]
        {
            let split_factors =
                par_givens && m * n >= 4096 && givens_b.len() * (m + n) >= 128 * 1024;
            if !split_factors {
                for &(k, cl, sl, cr, sr) in &givens_b {
                    rotate_rows(&mut u1t, k * m, (k + 1) * m, cl, sl, m);
                    rotate_rows(&mut v1, k * n, (k + 1) * n, cr, sr, n);
                }
            } else {
                std::thread::scope(|scope| {
                    let hu = scope.spawn(|| {
                        for &(k, cl, sl, _, _) in &givens_b {
                            rotate_rows(&mut u1t, k * m, (k + 1) * m, cl, sl, m);
                        }
                    });
                    let hv = scope.spawn(|| {
                        for &(k, _, _, cr, sr) in &givens_b {
                            rotate_rows(&mut v1, k * n, (k + 1) * n, cr, sr, n);
                        }
                    });
                    hu.join().unwrap();
                    hv.join().unwrap();
                });
            }
        }
        let (ub, sb, vtb) = svd_postprocess(&u1t, &v1, &sigma_b, &d_final, m, n, 1);
        u[b * m * n..(b + 1) * m * n].copy_from_slice(&ub);
        sigma[b * n..(b + 1) * n].copy_from_slice(&sb);
        vt[b * n * n..(b + 1) * n * n].copy_from_slice(&vtb);
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

/// LAPACK dbdsqr-style shifted QR iteration on an upper bidiagonal matrix
/// (main diagonal `d`, superdiagonal `e`). Returns the singular values, the
/// final diagonal (signs are absorbed into U by the caller) and the Givens
/// rotations `(k, cosl, sinl, cosr, sinr)` in application order so the
/// caller can rebuild the singular vectors.
pub(crate) fn dbdsqr_host<F: Float + Copy>(
    d: &[F],
    e: &[F],
    max_sweeps: usize,
) -> (Vec<F>, Vec<F>, Vec<GivensRotation<F>>) {
    let mut d = d.to_vec();
    let mut e = e.to_vec();
    let mut givens: Vec<GivensRotation<F>> = Vec::new();
    let sigma = dbdsqr(&mut d, &mut e, &mut givens, max_sweeps);
    (sigma, d, givens)
}

/// Sort the singular values descending, permute the factors and absorb the
/// diagonal signs into U.
///
/// Inputs are the transposed factor buffers (columns of U1 as rows of
/// `u1t`, rows of V1 as rows of `v1t`), the raw diagonal `d` from dbdsqr
/// (signs are absorbed into U) and the singular values. Output is the final
/// row-major `(u, sigma, vt)` in the `linalg::svd` layout (the wide-input
/// swap is applied by the caller), with numerical zeros masked relative to
/// `10 * eps * sigma_max`.
pub(crate) fn svd_postprocess<F: Float + Copy>(
    u1t: &[F],
    v1t: &[F],
    sigma_in: &[F],
    d: &[F],
    m: usize,
    n: usize,
    batch: usize,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    let mut u = vec![F::zero(); batch * m * n];
    let mut sigma = sigma_in.to_vec();
    let mut vt = vec![F::zero(); batch * n * n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut sorted = vec![F::zero(); n];
    for b in 0..batch {
        let u1t_b = &u1t[b * m * n..(b + 1) * m * n];
        let v1t_b = &v1t[b * n * n..(b + 1) * n * n];
        let sigma_b = &sigma[b * n..(b + 1) * n];
        let d_b = &d[b * n..(b + 1) * n];
        let smax = sigma_b.iter().fold(F::zero(), |s, &x| s.max(x.abs()));
        let zero_tol = smax * (F::epsilon() * F::from(10.0).unwrap());
        order.clear();
        order.extend(0..n);
        // Indices are compared by value only, so stability is irrelevant;
        // sort_unstable avoids the O(n) scratch allocation of stable sort.
        order.sort_unstable_by(|&i, &j| {
            sigma_b[j]
                .partial_cmp(&sigma_b[i])
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        // Apply the permutation while copying: in the transposed layout the
        // column permutation of U is a row permutation of U1t, and the row
        // permutation of Vt is a row permutation of V1t, so the factors land
        // in the output tensors directly (no intermediate permute buffers).
        // Negative diagonal entries (the dbdsqr sign convention) are
        // absorbed into U.
        for (t, &src) in order.iter().enumerate() {
            let flip = d_b[src] < F::zero();
            for i in 0..m {
                u[b * m * n + i * n + t] = if flip {
                    -u1t_b[src * m + i]
                } else {
                    u1t_b[src * m + i]
                };
            }
            for i in 0..n {
                vt[b * n * n + t * n + i] = v1t_b[src * n + i];
            }
            let v = sigma_b[src];
            sorted[t] = if v.abs() <= zero_tol { F::zero() } else { v };
        }
        for i in 0..n {
            sigma[b * n + i] = sorted[i];
        }
    }
    (u, sigma, vt)
}

/// Scaled Euclidean norm `sqrt(x^2 + y^2)` without overflow: rescale by the
/// larger input first so the squares stay finite for entries up to f32::MAX
/// (a plain `x*x + y*y` overflows past ~1.8e19).
#[inline]
fn scaled_norm2<F: Float + Copy>(x: F, y: F) -> F {
    let scale = x.abs().max(y.abs());
    if scale == F::zero() {
        F::zero()
    } else {
        let (u, v) = (x / scale, y / scale);
        scale * (u * u + v * v).sqrt()
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
    let s0 = scaled_norm2(e2, f2);
    let s1 = scaled_norm2(g2, h2);
    // Degenerate rows (σ at or below the same 10*eps*σmax tolerance used for
    // the sigma mask): a zero row is replaced by the surviving row's
    // orthogonal complement (or a unit basis when both rows are zero), so
    // Vt stays orthonormal even for rank-deficient inputs. The tolerance
    // matters: for exact rank deficiency the zero row is a transcendental
    // residual (~eps*σmax), never exactly 0.
    let zero_tol = s0.max(s1) * (F::epsilon() * F::from(10.0).unwrap());
    let (r0c, r0s) = if s0 > zero_tol {
        (e2 / s0, f2 / s0)
    } else {
        (F::zero(), F::zero())
    };
    let (r1c, r1s) = if s1 > zero_tol {
        (g2 / s1, h2 / s1)
    } else {
        (F::zero(), F::zero())
    };
    // Degenerate rows are replaced by the other row's orthogonal complement
    // (or a unit basis when both rows are zero). s0 >= s1 always holds, so a
    // degenerate first row implies a degenerate second row.
    let (r0c, r0s, r1c, r1s) = if s1 <= zero_tol {
        if s0 > zero_tol {
            (r0c, r0s, -r0s, r0c)
        } else {
            (F::one(), F::zero(), F::zero(), F::one())
        }
    } else {
        (r0c, r0s, r1c, r1s)
    };
    // Vt = diag(1/s0, 1/s1) UᵀA is already the exact right factor: its rows
    // are orthonormal (orthogonal rows, unit norms), and det(Vt) = +/-1 carries
    // the sign of det(A). Both signs are valid SVDs, so no handedness fix is
    // needed (a previous det < 0 negation of row 1 broke reconstruction).
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
fn bidiag_host<F: Float + Copy + core::ops::AddAssign>(
    a: &[F],
    m: usize,
    n: usize,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    // V1 kept in transposed layout (v1t[j*n+i] = V1[i,j]): the reflection
    // sweeps below then run over contiguous rows, and svd_host_seq consumes
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
                s += t * t;
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
            for v in wta[i..n].iter_mut() {
                *v = F::zero();
            }
            for k in i..m {
                slice_axpy(&mut wta[i..n], &a[k * n + i..k * n + n], w[k]);
            }
            let t = tau;
            for k in i..m {
                slice_axpy(&mut a[k * n + i..k * n + n], &wta[i..n], -(w[k] * t));
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
                    s += t * t;
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
                    aw[k] = slice_dot(&a[k * n..], &w, i + 1, n);
                }
                let t = tau;
                for k in i..m {
                    slice_axpy(&mut a[k * n + i + 1..k * n + n], &w[i + 1..n], -(aw[k] * t));
                }
                // V1 = V1 (I - tau w w^T) on the transposed factor: v1t[j, i2]
                // -= tau w[j] vw[i2] with vw[i2] = sum_j v1t[j, i2] w[j].
                // Both sweeps are outer-j/inner-i2, so they stream contiguous
                // rows and vectorize (accumulators live in the vw array, no
                // scalar reduction).
                for (j, &wj) in w[i + 1..n].iter().enumerate() {
                    let row = (i + 1 + j) * n;
                    slice_axpy(&mut vw[..n], &v1[row..row + n], wj);
                }
                for (j, &wj) in w[i + 1..n].iter().enumerate() {
                    let row = (i + 1 + j) * n;
                    slice_axpy(&mut v1[row..row + n], &vw[..n], -(tau * wj));
                }
                vw.fill(F::zero());
            }
        }
    }
    let mut u1 = vec![F::zero(); m * n];
    for i in 0..n {
        u1[i * n + i] = F::one();
    }
    // uw is reused across reflectors (zeroed per step), not reallocated.
    let mut uw = vec![F::zero(); n];
    for i in (0..n).rev() {
        let tau = taus[i];
        if tau != F::zero() {
            // uw[j] = w^T u1[:, j] computed over contiguous column strips:
            // u1[k, :] is row-major, so sweep k outer, j inner.
            uw.fill(F::zero());
            for k in i..m {
                slice_axpy(&mut uw[..n], &u1[k * n..k * n + n], ws[i * m + k]);
            }
            let t = tau;
            for k in i..m {
                slice_axpy(&mut u1[k * n..k * n + n], &uw[..n], -(ws[i * m + k] * t));
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
    givens: &mut Vec<GivensRotation<F>>,
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
        // Wilkinson-style shift from the bottom 2x2 block of B^T B. LAPACK
        // dbdsqr uses the SMALLER root (DLAS2 SSMIN) as the shift: the larger
        // root also converges, but loses relative accuracy on smaller singular
        // values at extreme dynamic range (errors scale with eps * smax, not
        // eps * sigma, for sigma far below smax). For f32 inputs the closed
        // form carries ~1e-5 error, which exceeds the deflation threshold
        // (10 eps) and stalls 2x2 blocks on rounding boundaries; computing
        // the shift in f64 keeps it exact.
        let shift = if core::mem::size_of::<F>() == 4 {
            let d1 = d[m - 2].to_f64().unwrap();
            let e1 = e[m - 2].to_f64().unwrap();
            let d2 = d[m - 1].to_f64().unwrap();
            let t = d1 * d1 + d2 * d2 + e1 * e1;
            let disc = (t * t - 4.0 * d1 * d1 * d2 * d2).max(0.0).sqrt();
            let smax = ((t + disc) / 2.0).max(0.0).sqrt();
            let smin = if smax > 0.0 {
                // |d1 d2| / smax, factored so the product cannot overflow.
                d1.abs() * (d2.abs() / smax)
            } else {
                0.0
            };
            F::from(smin).unwrap()
        } else {
            dlas2_smin(d[m - 2], e[m - 2], d[m - 1])
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
            // Identity rotations (sin == 0 on both sides, which happens for
            // deflated trailing entries) are exact no-ops for the factors;
            // skipping them keeps the application list shorter.
            if sl != F::zero() || sr != F::zero() {
                givens.push((i, cl, sl, cr, sr));
            }
        }
        e[m - 2] = f;
        iters += 1;
        if iters > max_sweeps * n {
            break;
        }
    }
    d.iter().map(|x| x.abs()).collect()
}

/// A single Givens rotation produced by [`dbdsqr_host`]: the pivot column
/// pair and the four rotation coefficients (cosl, sinl, cosr, sinr).
pub(crate) type GivensRotation<F> = (usize, F, F, F, F);

/// Smaller singular value of the 2x2 block [[d1, e1], [0, d2]], the LAPACK
/// dbdsqr shift. SSMIN = |d1 d2| / SSMAX is exact (the roots of the trailing
/// 2x2 of B^T B multiply to d1^2 d2^2), and dividing after scaling keeps both
/// the product and the quotient free of overflow/underflow.
fn dlas2_smin<F: Float + Copy>(d1: F, e1: F, d2: F) -> F {
    let smax = dlas2_smax(d1, e1, d2);
    if smax == F::zero() {
        F::zero()
    } else {
        (d1 * (d2 / smax)).abs()
    }
}

/// Largest singular value of the 2x2 block [[d1, e1], [0, d2]]. Scaled like
/// LAPACK dlas2: no overflow or underflow on the intermediate squares.
fn dlas2_smax<F: Float + Copy>(d1: F, e1: F, d2: F) -> F {
    let t = d1 * d1 + d2 * d2 + e1 * e1;
    let disc = t * t - F::from(4.0).unwrap() * d1 * d1 * d2 * d2;
    // Fast path only when the discriminant is computable: for f64 entries in
    // ~[7.7e76, 1.3e154], t is finite but t*t overflows, and max(NaN, 0) = 0
    // would silently return a wrong (low or inf) result. disc < 0 is normal
    // (rounding) and clamps to 0; only non-finite values need the scaled path.
    if t.is_finite() && disc.is_finite() {
        ((t + disc.max(F::zero()).sqrt()) / F::from(2.0).unwrap())
            .max(F::zero())
            .sqrt()
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

    /// Deterministic LCG in [0, 1), shared by the pipeline tests.
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32) / (1u32 << 31) as f32
        }
    }

    /// The AVX2 kernels and the portable fallback must produce identical
    /// buffers: both paths run the same per-element operations in the same
    /// order, and this assertion pins that contract. Each shape exercises a
    /// different kernel subset: 96x48 covers axpy/dot/rotate, the n = 1
    /// shapes cover the scaled sum of squares, and the last case covers the
    /// f64 bodies.
    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    fn test_svd_simd_paths_bit_identical() {
        use std::sync::atomic::Ordering;

        if is_x86_feature_detected!("avx2") {
            // Forcing the AVX2 path without the hardware would fault; on
            // such machines every run already uses the fallback.
            let mut next = lcg(0x5EED_1234_ABCD_EF01);
            let tall: Vec<f32> = (0..96 * 48).map(|_| next() * 2.0 - 1.0).collect();
            let col: Vec<f32> = (0..12).map(|_| next() * 2.0 - 1.0).collect();
            let col_f64: Vec<f64> = (0..10).map(|_| next() as f64 * 0.4 - 0.2).collect();

            macro_rules! compare_paths {
                ($a:expr, $m:expr, $n:expr, $t:ty) => {{
                    SIMD_TEST_OVERRIDE.store(1, Ordering::Relaxed); // force AVX2
                    let avx2 = svd_host::<$t>($a, $m, $n, 1, 15, false);
                    SIMD_TEST_OVERRIDE.store(2, Ordering::Relaxed); // force fallback
                    let scalar = svd_host::<$t>($a, $m, $n, 1, 15, false);
                    SIMD_TEST_OVERRIDE.store(0, Ordering::Relaxed);
                    assert_eq!(avx2.0, scalar.0, "U differs");
                    assert_eq!(avx2.1, scalar.1, "sigma differs");
                    assert_eq!(avx2.2, scalar.2, "Vt differs");
                }};
            }
            compare_paths!(&tall[..], 96, 48, f32);
            compare_paths!(&col[..], 12, 1, f32);
            compare_paths!(&col_f64[..], 10, 1, f64);
        }
    }

    /// Defensive numeric paths of the LAPACK-style helpers, driven directly:
    /// all-zero and overflow-scale 2x2 shifts, underflowing Givens
    /// generation, and the sweep budget bail-out.
    #[test]
    fn test_svd_host_numeric_guards() {
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
        // dbdsqr_host: a zero sweep budget stops after one pass instead of
        // looping forever; results stay finite.
        let (sigma, _d, _givens) = dbdsqr_host(&[3.0f64, 1.0], &[0.5], 0);
        assert!(sigma.iter().all(|x| x.is_finite()), "{sigma:?}");
    }

    fn recon_err<F: Float + Copy>(u: &[F], s: &[F], vt: &[F], a: &[F], m: usize, n: usize) -> F {
        assert_eq!(
            (u.len(), s.len(), vt.len(), a.len()),
            (m * n, n, n * n, m * n)
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
        assert!((s[0] - 25.46240743603639).abs() < 1e-12);
        assert!((s[1] - 1.290661675761233).abs() < 1e-12);
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
        assert!(s[0].is_finite() && (s[0] - 1e200).abs() < 1e200 * 1e-14);
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

        // Negative determinant: the right factor must still reconstruct A.
        // (A regression test: a previous "handedness fix" negated Vt row 1
        // without U column 1, breaking every det < 0 input.)
        for a in [
            [1.0f64, 2.0, 3.0, 4.0],
            [0.0f64, 1.0, 1.0, 0.0],
            [-3.0f64, 1.0, 2.0, -1.0],
        ] {
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
            assert!(err < 1e-12, "neg-det 2x2 recon {err} for {a:?}");
        }
    }

    #[test]
    fn test_svd_host_2x2_rank_deficient_orthonormal() {
        // Rank-deficient 2x2: the zero singular value comes out as a
        // transcendental residual (~eps * smax), never exactly 0; the
        // fallback must still replace the degenerate Vt row with an
        // orthonormal complement (the 10*eps*smax tolerance, same as the
        // sigma mask).
        for a in [
            [1.0f64, 2.0, 2.0, 4.0],
            [0.0f64, 1.0, 0.0, 0.0],
            [0.0f64, 0.0, 1.0, 0.0],
            [1.0f64, 1.0, 2.0, 2.0],
        ] {
            let (u, s, vt) = svd_host::<f64>(&a, 2, 2, 1, 30, false);
            assert_eq!(s[1], 0.0, "masked sigma for {a:?}");
            // Vt rows orthonormal: Vt Vt^T = I.
            let mut ortho = 0.0f64;
            for i in 0..2 {
                for j in 0..2 {
                    let mut acc = 0.0f64;
                    for k in 0..2 {
                        acc += vt[i * 2 + k] * vt[j * 2 + k];
                    }
                    ortho = ortho.max((acc - if i == j { 1.0 } else { 0.0 }).abs());
                }
            }
            assert!(ortho < 1e-12, "Vt orthonormal for {a:?}, err {ortho}");
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
            assert!(err < 1e-12, "rank-def 2x2 recon {err} for {a:?}");
        }
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

    #[test]
    fn test_svd_host_zero_m1_orthonormal() {
        // Zero m x 1 matrix: U must stay orthonormal (unit basis), not an
        // all-ones column of norm sqrt(m).
        let (m, n) = (5usize, 1usize);
        let a = [0.0f64; 5];
        let (u, s, vt) = svd_host::<f64>(&a, m, n, 1, 30, false);
        assert_eq!(s[0], 0.0);
        let norm: f64 = u.iter().map(|x| x * x).sum();
        assert!((norm - 1.0).abs() < 1e-15, "U column norm {norm}");
        assert_eq!(vt[0], 1.0);
        // Non-zero column keeps the normalized-column form.
        let a = [0.0f64, 3.0, 0.0, 4.0, 0.0];
        let (u, s, vt) = svd_host::<f64>(&a, m, n, 1, 30, false);
        assert!((s[0] - 5.0).abs() < 1e-15);
        let mut err = 0.0f64;
        for i in 0..m {
            err = err.max((a[i] - u[i * n] * s[0] * vt[0]).abs());
        }
        assert!(err < 1e-15, "m1 recon {err}");
    }
}
