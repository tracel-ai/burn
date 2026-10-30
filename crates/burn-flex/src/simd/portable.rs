//! Portable SIMD kernels using macerator.
//!
//! Replaces platform-specific implementations (neon.rs) with a single
//! portable implementation that auto-dispatches to NEON/AVX2/SSE/SIMD128/scalar.

use alloc::vec::Vec;
use core::mem::MaybeUninit;

use macerator::{
    Mask, Scalar, Simd, VBitAnd, VBitOr, VBitXor, VCeil, VFloor, VRoundTiesEven, VTrunc, Vector,
    vload_unaligned, vstore_unaligned,
};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Threshold for parallel execution (elements).
/// For memory-bound operations, parallelism helps when data exceeds L3 cache.
#[cfg(feature = "rayon")]
const PARALLEL_THRESHOLD: usize = 4 * 1024 * 1024;

#[cfg(feature = "rayon")]
const CHUNK_SIZE: usize = 4096;

// ============================================================================
// f32 in-place binary ops
// ============================================================================

// The op is written as a closure-shaped expression `|a, b| ...` where
// `a` is the destination element and `b` the source element. This lets
// one macro stamp out both the natural kernels (`a - b`) and the
// reversed ones (`b - a`) needed when operands are dispatched swapped.
macro_rules! define_inplace_f32_op {
    ($pub_fn:ident, $seq_fn:ident, $par_fn:ident, |$a:ident, $b:ident| $body:expr) => {
        #[inline]
        pub fn $pub_fn(a: &mut [f32], b: &[f32]) {
            debug_assert_eq!(a.len(), b.len());

            #[cfg(feature = "rayon")]
            if a.len() >= PARALLEL_THRESHOLD {
                $par_fn(a, b);
                return;
            }

            $seq_fn(a, b);
        }

        #[macerator::with_simd]
        #[allow(clippy::assign_op_pattern)]
        fn $seq_fn<S: Simd>(a: &mut [f32], b: &[f32]) {
            let lanes = S::lanes32();
            let len = a.len();
            let simd_len = len / lanes * lanes;

            let mut i = 0;
            while i < simd_len {
                unsafe {
                    let $a = vload_unaligned(a.as_ptr().add(i));
                    let $b = vload_unaligned(b.as_ptr().add(i));
                    vstore_unaligned::<S, _>(a.as_mut_ptr().add(i), $body);
                }
                i += lanes;
            }

            for j in simd_len..len {
                let $a = a[j];
                let $b = b[j];
                a[j] = $body;
            }
        }

        #[cfg(feature = "rayon")]
        fn $par_fn(a: &mut [f32], b: &[f32]) {
            a.par_chunks_mut(CHUNK_SIZE)
                .zip(b.par_chunks(CHUNK_SIZE))
                .for_each(|(a_chunk, b_chunk)| {
                    $seq_fn(a_chunk, b_chunk);
                });
        }
    };
}

define_inplace_f32_op!(
    add_inplace_f32,
    add_inplace_f32_seq,
    add_inplace_f32_par,
    |a, b| a + b
);
define_inplace_f32_op!(
    sub_inplace_f32,
    sub_inplace_f32_seq,
    sub_inplace_f32_par,
    |a, b| a - b
);
define_inplace_f32_op!(
    mul_inplace_f32,
    mul_inplace_f32_seq,
    mul_inplace_f32_par,
    |a, b| a * b
);
define_inplace_f32_op!(
    div_inplace_f32,
    div_inplace_f32_seq,
    div_inplace_f32_par,
    |a, b| a / b
);

// Reversed kernels: `a[i] = b[i] OP a[i]`. Sub and div are not
// commutative, so when a binary op is dispatched with swapped operands
// (the destination buffer holds the original *rhs*), the kernel must
// compute `src OP dst` instead of `dst OP src`.
define_inplace_f32_op!(
    rsub_inplace_f32,
    rsub_inplace_f32_seq,
    rsub_inplace_f32_par,
    |a, b| b - a
);
define_inplace_f32_op!(
    rdiv_inplace_f32,
    rdiv_inplace_f32_seq,
    rdiv_inplace_f32_par,
    |a, b| b / a
);

// ============================================================================
// f32 in-place unary ops (SIMD-accelerated)
// ============================================================================

#[inline]
pub fn abs_inplace_f32(a: &mut [f32]) {
    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        abs_inplace_f32_par(a);
        return;
    }

    abs_inplace_f32_seq(a);
}

#[macerator::with_simd]
fn abs_inplace_f32_seq<S: Simd>(a: &mut [f32]) {
    let lanes = S::lanes32();
    let len = a.len();
    let simd_len = len / lanes * lanes;

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let v = vload_unaligned::<S, _>(a.as_ptr().add(i));
            vstore_unaligned::<S, _>(a.as_mut_ptr().add(i), v.abs());
        }
        i += lanes;
    }

    for v in &mut a[simd_len..len] {
        *v = v.abs();
    }
}

#[cfg(feature = "rayon")]
fn abs_inplace_f32_par(a: &mut [f32]) {
    a.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
        abs_inplace_f32_seq(chunk);
    });
}

#[inline]
pub fn recip_inplace_f32(a: &mut [f32]) {
    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        recip_inplace_f32_par(a);
        return;
    }

    recip_inplace_f32_seq(a);
}

#[macerator::with_simd]
fn recip_inplace_f32_seq<S: Simd>(a: &mut [f32]) {
    let lanes = S::lanes32();
    let len = a.len();
    let simd_len = len / lanes * lanes;

    // Use exact SIMD division (not VRecip which is approximate on NEON/SSE)
    let ones = 1.0f32.splat::<S>();

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let v = vload_unaligned::<S, _>(a.as_ptr().add(i));
            vstore_unaligned::<S, _>(a.as_mut_ptr().add(i), ones / v);
        }
        i += lanes;
    }

    for v in &mut a[simd_len..len] {
        *v = 1.0 / *v;
    }
}

#[cfg(feature = "rayon")]
fn recip_inplace_f32_par(a: &mut [f32]) {
    a.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
        recip_inplace_f32_seq(chunk);
    });
}

// ============================================================================
// Lane-wise maps
// ============================================================================

/// Lanes in the widest register of the smallest element: 64 `u8` lanes in
/// 512 bits. Sizes the stack buffers that run a short tail through a vector.
pub const MAX_LANES: usize = 64;

/// An elementwise op on vectors, run over slices by [`map_inplace`] and
/// [`map_to_vec`]. Each output lane must depend only on the same input lane,
/// since the drivers also apply it to the padding lanes of a short tail.
pub trait LaneOp<F: Scalar>: Copy + Send + Sync {
    fn apply<S: Simd>(self, v: Vector<S, F>) -> Vector<S, F>;
}

/// Applies `op` to every element of `a` in place.
#[inline]
pub fn map_inplace<F: Scalar + Send, O: LaneOp<F>>(a: &mut [F], op: O) {
    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        a.par_chunks_mut(CHUNK_SIZE)
            .for_each(|chunk| map_inplace_seq(chunk, op));
        return;
    }

    map_inplace_seq(a, op);
}

/// Returns `op` of each element of `src`, computed in one pass into a new
/// vector.
#[inline]
pub fn map_to_vec<F: Scalar + Send + Sync, O: LaneOp<F>>(src: &[F], op: O) -> Vec<F> {
    let mut out = Vec::with_capacity(src.len());
    let dst = &mut out.spare_capacity_mut()[..src.len()];

    #[cfg(feature = "rayon")]
    if src.len() >= PARALLEL_THRESHOLD {
        dst.par_chunks_mut(CHUNK_SIZE)
            .zip(src.par_chunks(CHUNK_SIZE))
            .for_each(|(d, s)| map_into_seq(s, d, op));
    } else {
        map_into_seq(src, dst, op);
    }
    #[cfg(not(feature = "rayon"))]
    map_into_seq(src, dst, op);

    // SAFETY: `map_into_seq` wrote every one of the `src.len()` elements.
    unsafe { out.set_len(src.len()) };
    out
}

#[macerator::with_simd]
fn map_inplace_seq<S: Simd, F: Scalar, O: LaneOp<F>>(a: &mut [F], op: O) {
    let p = a.as_mut_ptr();
    // SAFETY: `p` is valid for `a.len()` reads and writes.
    unsafe { map_raw::<S, F, O>(p, p, a.len(), op) }
}

#[macerator::with_simd]
fn map_into_seq<S: Simd, F: Scalar, O: LaneOp<F>>(src: &[F], dst: &mut [MaybeUninit<F>], op: O) {
    assert_eq!(src.len(), dst.len());
    // SAFETY: `src` and `dst` are distinct borrows of `src.len()` elements, and
    // `MaybeUninit<F>` has the layout of `F`.
    unsafe { map_raw::<S, F, O>(src.as_ptr(), dst.as_mut_ptr().cast(), src.len(), op) }
}

/// Writes `op(src[i])` to `dst[i]` for every `i < len`. The tail shorter than
/// a vector goes through `op` in a padded buffer, so each element gets the
/// same result wherever it sits in the slice.
///
/// # Safety
///
/// `src` must be valid for `len` reads and `dst` for `len` writes, both
/// aligned to `F`: the vector loads and stores are unaligned, but the tail
/// copies use `copy_nonoverlapping`. They are either equal or don't overlap.
#[inline(always)]
unsafe fn map_raw<S: Simd, F: Scalar, O: LaneOp<F>>(src: *const F, dst: *mut F, len: usize, op: O) {
    let lanes = F::lanes::<S>();
    let simd_len = len / lanes * lanes;
    let unrolled_len = len / (4 * lanes) * (4 * lanes);

    // Four independent vectors per iteration, to match LLVM's unrolled
    // autovectorized loops on the cheap ops and overlap the long ones. The
    // loads come first because `src` and `dst` may alias, so the compiler
    // can't move later loads above earlier stores.
    let mut i = 0;
    while i < unrolled_len {
        unsafe {
            let v0 = vload_unaligned::<S, F>(src.add(i));
            let v1 = vload_unaligned::<S, F>(src.add(i + lanes));
            let v2 = vload_unaligned::<S, F>(src.add(i + 2 * lanes));
            let v3 = vload_unaligned::<S, F>(src.add(i + 3 * lanes));
            vstore_unaligned::<S, F>(dst.add(i), op.apply(v0));
            vstore_unaligned::<S, F>(dst.add(i + lanes), op.apply(v1));
            vstore_unaligned::<S, F>(dst.add(i + 2 * lanes), op.apply(v2));
            vstore_unaligned::<S, F>(dst.add(i + 3 * lanes), op.apply(v3));
        }
        i += 4 * lanes;
    }
    while i < simd_len {
        unsafe {
            let v = vload_unaligned::<S, F>(src.add(i));
            vstore_unaligned::<S, F>(dst.add(i), op.apply(v));
        }
        i += lanes;
    }

    let tail = len - simd_len;
    if tail > 0 {
        let mut buf = [F::default(); MAX_LANES];
        assert!(
            lanes <= MAX_LANES,
            "SIMD lane count {lanes} exceeds {MAX_LANES}"
        );
        // SAFETY: the tail is in bounds of both buffers, and the assertion
        // above keeps one vector within `buf`.
        unsafe {
            core::ptr::copy_nonoverlapping(src.add(simd_len), buf.as_mut_ptr(), tail);
            let v = vload_unaligned::<S, F>(buf.as_ptr());
            vstore_unaligned::<S, F>(buf.as_mut_ptr(), op.apply(v));
            core::ptr::copy_nonoverlapping(buf.as_ptr(), dst.add(simd_len), tail);
        }
    }
}

/// An elementwise op on two vectors, run over slice pairs by
/// [`zip_map_inplace`] and [`zip_map_to_vec`]. Like [`LaneOp`], each output
/// lane must depend only on the same lane of each input.
pub trait LaneOp2<F: Scalar>: Copy + Send + Sync {
    fn apply<S: Simd>(self, a: Vector<S, F>, b: Vector<S, F>) -> Vector<S, F>;
}

/// `O` with its operands swapped, so a result can be written into either.
#[derive(Clone, Copy)]
pub struct Swap<O>(pub O);

impl<F: Scalar, O: LaneOp2<F>> LaneOp2<F> for Swap<O> {
    #[inline(always)]
    fn apply<S: Simd>(self, a: Vector<S, F>, b: Vector<S, F>) -> Vector<S, F> {
        self.0.apply(b, a)
    }
}

/// Sets `a[i] = op(a[i], b[i])` for every `i`.
#[inline]
pub fn zip_map_inplace<F: Scalar + Send + Sync, O: LaneOp2<F>>(a: &mut [F], b: &[F], op: O) {
    assert_eq!(a.len(), b.len());
    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        a.par_chunks_mut(CHUNK_SIZE)
            .zip(b.par_chunks(CHUNK_SIZE))
            .for_each(|(a, b)| zip_inplace_seq(a, b, op));
        return;
    }

    zip_inplace_seq(a, b, op);
}

/// Returns `op(a[i], b[i])` for every `i`, computed in one pass into a new
/// vector.
#[inline]
pub fn zip_map_to_vec<F: Scalar + Send + Sync, O: LaneOp2<F>>(a: &[F], b: &[F], op: O) -> Vec<F> {
    assert_eq!(a.len(), b.len());
    let mut out = Vec::with_capacity(a.len());
    let dst = &mut out.spare_capacity_mut()[..a.len()];

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        dst.par_chunks_mut(CHUNK_SIZE)
            .zip(a.par_chunks(CHUNK_SIZE).zip(b.par_chunks(CHUNK_SIZE)))
            .for_each(|(d, (a, b))| zip_into_seq(a, b, d, op));
    } else {
        zip_into_seq(a, b, dst, op);
    }
    #[cfg(not(feature = "rayon"))]
    zip_into_seq(a, b, dst, op);

    // SAFETY: `zip_into_seq` wrote every one of the `a.len()` elements.
    unsafe { out.set_len(a.len()) };
    out
}

#[macerator::with_simd]
fn zip_inplace_seq<S: Simd, F: Scalar, O: LaneOp2<F>>(a: &mut [F], b: &[F], op: O) {
    let p = a.as_mut_ptr();
    // SAFETY: `p` is valid for `a.len()` reads and writes and `b` for as many
    // reads.
    unsafe { zip_raw::<S, F, O>(p, b.as_ptr(), p, a.len(), op) }
}

#[macerator::with_simd]
fn zip_into_seq<S: Simd, F: Scalar, O: LaneOp2<F>>(
    a: &[F],
    b: &[F],
    dst: &mut [MaybeUninit<F>],
    op: O,
) {
    // SAFETY: `dst` is a distinct borrow of `a.len()` elements, and
    // `MaybeUninit<F>` has the layout of `F`.
    unsafe { zip_raw::<S, F, O>(a.as_ptr(), b.as_ptr(), dst.as_mut_ptr().cast(), a.len(), op) }
}

/// Writes `op(a[i], b[i])` to `dst[i]` for every `i < len`, with a short tail
/// through `op` in padded buffers as in [`map_raw`].
///
/// # Safety
///
/// `a` and `b` must be valid for `len` reads and `dst` for `len` writes, all
/// aligned to `F`. `dst` either equals `a` or overlaps neither input.
#[inline(always)]
unsafe fn zip_raw<S: Simd, F: Scalar, O: LaneOp2<F>>(
    a: *const F,
    b: *const F,
    dst: *mut F,
    len: usize,
    op: O,
) {
    let lanes = F::lanes::<S>();
    let simd_len = len / lanes * lanes;
    let unrolled_len = len / (4 * lanes) * (4 * lanes);

    // Four vectors per iteration with the loads first, as in `map_raw`.
    let mut i = 0;
    while i < unrolled_len {
        unsafe {
            let a0 = vload_unaligned::<S, F>(a.add(i));
            let a1 = vload_unaligned::<S, F>(a.add(i + lanes));
            let a2 = vload_unaligned::<S, F>(a.add(i + 2 * lanes));
            let a3 = vload_unaligned::<S, F>(a.add(i + 3 * lanes));
            let b0 = vload_unaligned::<S, F>(b.add(i));
            let b1 = vload_unaligned::<S, F>(b.add(i + lanes));
            let b2 = vload_unaligned::<S, F>(b.add(i + 2 * lanes));
            let b3 = vload_unaligned::<S, F>(b.add(i + 3 * lanes));
            vstore_unaligned::<S, F>(dst.add(i), op.apply(a0, b0));
            vstore_unaligned::<S, F>(dst.add(i + lanes), op.apply(a1, b1));
            vstore_unaligned::<S, F>(dst.add(i + 2 * lanes), op.apply(a2, b2));
            vstore_unaligned::<S, F>(dst.add(i + 3 * lanes), op.apply(a3, b3));
        }
        i += 4 * lanes;
    }
    while i < simd_len {
        unsafe {
            let va = vload_unaligned::<S, F>(a.add(i));
            let vb = vload_unaligned::<S, F>(b.add(i));
            vstore_unaligned::<S, F>(dst.add(i), op.apply(va, vb));
        }
        i += lanes;
    }

    let tail = len - simd_len;
    if tail > 0 {
        let mut buf_a = [F::default(); MAX_LANES];
        let mut buf_b = [F::default(); MAX_LANES];
        assert!(
            lanes <= MAX_LANES,
            "SIMD lane count {lanes} exceeds {MAX_LANES}"
        );
        // SAFETY: the tail is in bounds of every buffer, and the assertion
        // above keeps one vector within `buf_a` and `buf_b`.
        unsafe {
            core::ptr::copy_nonoverlapping(a.add(simd_len), buf_a.as_mut_ptr(), tail);
            core::ptr::copy_nonoverlapping(b.add(simd_len), buf_b.as_mut_ptr(), tail);
            let va = vload_unaligned::<S, F>(buf_a.as_ptr());
            let vb = vload_unaligned::<S, F>(buf_b.as_ptr());
            vstore_unaligned::<S, F>(buf_a.as_mut_ptr(), op.apply(va, vb));
            core::ptr::copy_nonoverlapping(buf_a.as_ptr(), dst.add(simd_len), tail);
        }
    }
}

// Rounding is exact, so these match std's `floor`, `ceil`, `trunc` and
// `round_ties_even` bit for bit, signed zeros included.

#[derive(Clone, Copy)]
pub struct Floor;
#[derive(Clone, Copy)]
pub struct Ceil;
#[derive(Clone, Copy)]
pub struct Trunc;
/// Nearest integer, ties to even.
#[derive(Clone, Copy)]
pub struct RoundTiesEven;

impl<F: VFloor> LaneOp<F> for Floor {
    #[inline(always)]
    fn apply<S: Simd>(self, v: Vector<S, F>) -> Vector<S, F> {
        v.floor()
    }
}

impl<F: VCeil> LaneOp<F> for Ceil {
    #[inline(always)]
    fn apply<S: Simd>(self, v: Vector<S, F>) -> Vector<S, F> {
        v.ceil()
    }
}

impl<F: VTrunc> LaneOp<F> for Trunc {
    #[inline(always)]
    fn apply<S: Simd>(self, v: Vector<S, F>) -> Vector<S, F> {
        v.trunc()
    }
}

impl<F: VRoundTiesEven> LaneOp<F> for RoundTiesEven {
    #[inline(always)]
    fn apply<S: Simd>(self, v: Vector<S, F>) -> Vector<S, F> {
        v.round_ties_even()
    }
}

/// `e^x` for f32 lanes: Cody-Waite reduction to `|r|` of about `ln(2) / 2` and
/// the Cephes `expf` polynomial. Within 2 ulp of libm over the whole range,
/// subnormal results included. Results below the smallest subnormal give 0,
/// results that overflow give +inf, and NaN passes through.
#[inline(always)]
pub fn exp_f32<S: Simd>(x: Vector<S, f32>) -> Vector<S, f32> {
    // exp(-104) rounds to 0 and exp(89) overflows, so clamping changes no
    // result and bounds `n` for the exponent bits below.
    let xc = x.max((-104.0f32).splat()).min(89.0f32.splat());
    // Adding 1.5 * 2^23 rounds `x * log2(e)` to an integer, ties to even,
    // and leaves it in the low mantissa bits. Cheaper than `round_to_int`,
    // which pays for saturation the clamp already rules out.
    let magic = 12_582_912.0f32.splat::<S>();
    let t = xc * core::f32::consts::LOG2_E.splat() + magic;
    let nf = t - magic;
    let n = t.reinterpret::<i32>() - magic.reinterpret::<i32>();
    // ln(2) split in two; the high part has 9 significant bits, so the first
    // product is exact for the clamped `n`.
    let r = xc - nf * 0.693_359_4f32.splat() - nf * (-2.121_944_4e-4f32).splat();

    let mut p = 1.987_569_1e-4f32.splat::<S>();
    p = p.mul_add(r, 1.398_199_9e-3f32.splat());
    p = p.mul_add(r, 8.333_452e-3f32.splat());
    p = p.mul_add(r, 4.166_579_6e-2f32.splat());
    p = p.mul_add(r, 1.666_666_6e-1f32.splat());
    p = p.mul_add(r, 0.5f32.splat());
    let y = p.mul_add(r * r, r) + 1.0f32.splat();

    // 2^n as two factors keeps both biased exponents in range for every
    // clamped `n` (-150 for subnormals up to 128), and the last multiply
    // rounds once into the subnormal or overflow range.
    let n1 = n >> 1;
    let n2 = n - n1;
    let bias = 127i32.splat::<S>();
    let s1 = ((n1 + bias) << 23).reinterpret::<f32>();
    let s2 = ((n2 + bias) << 23).reinterpret::<f32>();
    let out = y * s1 * s2;

    // `min` and `max` return a backend-dependent operand for NaN.
    let is_nan = !x.le(x);
    is_nan.select(x, out)
}

/// [`exp_f32`] as a [`LaneOp`].
#[derive(Clone, Copy)]
pub struct Exp;

impl LaneOp<f32> for Exp {
    #[inline(always)]
    fn apply<S: Simd>(self, v: Vector<S, f32>) -> Vector<S, f32> {
        exp_f32::<S>(v)
    }
}

/// Logistic sigmoid for f32 lanes, built on `e = exp(-|x|)` so it never
/// overflows: `s = 1 / (1 + e)` for `x >= 0` and `e * s` below.
#[derive(Clone, Copy)]
pub struct Sigmoid;

impl LaneOp<f32> for Sigmoid {
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>) -> Vector<S, f32> {
        let e = exp_f32::<S>(0.0f32.splat::<S>() - x.abs());
        let s = 1.0f32.splat::<S>() / (1.0f32.splat() + e);
        x.ge(0.0f32.splat()).select(s, e * s)
    }
}

/// `ln(sigmoid(x))` for f32 lanes, as `-ln(1 + e^-x)` for `x >= 0` and
/// `x - ln(1 + e^x)` below, so the exponential never overflows.
#[derive(Clone, Copy)]
pub struct LogSigmoid;

impl LaneOp<f32> for LogSigmoid {
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>) -> Vector<S, f32> {
        let zero = 0.0f32.splat::<S>();
        let l = Ln1p.apply(exp_f32::<S>(zero - x.abs()));
        // `-1 * l` rather than `0 - l`, so an underflowed `l` gives -0.
        x.ge(zero).select((-1.0f32).splat() * l, x - l)
    }
}

/// The derivative of [`LogSigmoid`], `sigmoid(-x)`.
#[derive(Clone, Copy)]
pub struct LogSigmoidGrad;

impl LaneOp<f32> for LogSigmoidGrad {
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>) -> Vector<S, f32> {
        Sigmoid.apply(0.0f32.splat::<S>() - x)
    }
}

/// The polynomial with coefficients `c`, highest degree first, at `x`.
#[inline(always)]
fn horner<S: Simd, const N: usize>(x: Vector<S, f32>, c: [f32; N]) -> Vector<S, f32> {
    let mut p = c[0].splat::<S>();
    for &ci in &c[1..] {
        p = p.mul_add(x, ci.splat());
    }
    p
}

/// `mag` with the sign of `sign`, for a non-negative `mag`.
#[inline(always)]
fn with_sign_of<S: Simd>(mag: Vector<S, f32>, sign: Vector<S, f32>) -> Vector<S, f32> {
    let sign_bit = sign.reinterpret::<i32>() & i32::MIN.splat();
    (mag.reinterpret::<i32>() | sign_bit).reinterpret::<f32>()
}

// `Ln`, `Ln1p` and `erf_f32` are adapted from FreeBSD msun's `e_logf.c`,
// `s_log1pf.c` and `s_erff.c`, by way of the `libm` crate, under this notice:
//
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================

/// `ln(x)` for f32 lanes: the `libm` crate's `logf` (FreeBSD `e_logf.c`),
/// step for step and without fused multiply-adds, so it matches it bit for
/// bit. Negative inputs give NaN, zeros give -inf, and +inf and NaN pass
/// through.
#[derive(Clone, Copy)]
pub struct Ln;

impl LaneOp<f32> for Ln {
    #[allow(clippy::excessive_precision)]
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>) -> Vector<S, f32> {
        // Subnormals are scaled by 2^25 into the normal range.
        // `Mask` isn't `Copy`, so each select takes a fresh comparison.
        let sub = || x.lt(f32::MIN_POSITIVE.splat());
        let xs = sub().select(x * 33_554_432.0f32.splat(), x);
        let k_sub = sub().select((-25.0f32).splat(), 0.0f32.splat());

        // Splits x into 2^k * m with m in [sqrt(2)/2, sqrt(2)).
        let ix = xs.reinterpret::<i32>() + (0x3f80_0000 - 0x3f35_04f3).splat();
        let k = ((ix >> 23) - 0x7f.splat()).to_float() + k_sub;
        let m = ((ix & 0x007f_ffff.splat()) + 0x3f35_04f3.splat()).reinterpret::<f32>();

        let f = m - 1.0f32.splat();
        let s = f / (2.0f32.splat() + f);
        let z = s * s;
        let w = z * z;
        let t1 = w * (0.40000972152f32.splat() + w * 0.24279078841f32.splat());
        let t2 = z * (0.66666662693f32.splat() + w * 0.28498786688f32.splat());
        let r = t2 + t1;
        let hfsq = 0.5f32.splat() * f * f;
        let out = s * (hfsq + r) + k * 9.0580006145e-06f32.splat() - hfsq
            + f
            + k * 6.9313812256e-01f32.splat();

        let out = x.eq(0.0f32.splat()).select(f32::NEG_INFINITY.splat(), out);
        let out = x.lt(0.0f32.splat()).select(f32::NAN.splat(), out);
        (!x.lt(f32::INFINITY.splat())).select(x, out)
    }
}

/// `ln(1 + x)` for f32 lanes: the `libm` crate's `log1pf` (FreeBSD
/// `s_log1pf.c`), step for step and without fused multiply-adds, so it
/// matches it bit for bit. Both of its reductions are computed and selected
/// per lane. -1 gives -inf, inputs below -1 give NaN, and +inf and NaN pass
/// through.
#[derive(Clone, Copy)]
pub struct Ln1p;

impl LaneOp<f32> for Ln1p {
    #[allow(clippy::excessive_precision)]
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>) -> Vector<S, f32> {
        let one = 1.0f32.splat::<S>();
        let zero = 0.0f32.splat::<S>();

        // Near 0, where 1 + x is in [sqrt(2)/2, sqrt(2)), f = x and k = 0.
        // Elsewhere 1 + x is split into 2^k * m like `Ln`, and `c` corrects
        // for the rounding of 1 + x. `Mask` isn't `Copy`, so each select takes
        // a fresh comparison.
        let near = || {
            !(x.lt(f32::from_bits(0xbe95_f619).splat())
                | !x.lt(f32::from_bits(0x3ed4_13d0).splat()))
        };
        let u = one + x;
        let iu = u.reinterpret::<i32>() + (0x3f80_0000 - 0x3f35_04f3).splat();
        let k_far = ((iu >> 23) - 0x7f.splat()).to_float();
        let c_far = (!k_far.lt(2.0f32.splat())).select(one - (u - x), x - (u - one)) / u;
        let c_far = k_far.lt(25.0f32.splat()).select(c_far, zero);
        let f_far = ((iu & 0x007f_ffff.splat()) + 0x3f35_04f3.splat()).reinterpret::<f32>() - one;

        let f = near().select(x, f_far);
        let c = near().select(zero, c_far);
        let k = near().select(zero, k_far);

        let s = f / (2.0f32.splat() + f);
        let z = s * s;
        let w = z * z;
        let t1 = w * (0.40000972152f32.splat() + w * 0.24279078841f32.splat());
        let t2 = z * (0.66666662693f32.splat() + w * 0.28498786688f32.splat());
        let r = t2 + t1;
        let hfsq = 0.5f32.splat() * f * f;
        let out = s * (hfsq + r) + (k * 9.0580006145e-06f32.splat() + c) - hfsq
            + f
            + k * 6.9313812256e-01f32.splat();

        // |x| < 2^-24 gives x itself.
        let out = x
            .abs()
            .lt(f32::from_bits(0x3380_0000).splat())
            .select(x, out);
        let out = x
            .eq((-1.0f32).splat())
            .select(f32::NEG_INFINITY.splat(), out);
        let out = x.lt((-1.0f32).splat()).select(f32::NAN.splat(), out);
        (!x.lt(f32::INFINITY.splat())).select(x, out)
    }
}

/// `atanh(x)` for f32 lanes: the `libm` crate's `atanhf` (from musl) on
/// [`Ln1p`], so it matches it bit for bit. +-1 give +-inf and inputs beyond
/// them give NaN.
#[derive(Clone, Copy)]
pub struct Atanh;

impl LaneOp<f32> for Atanh {
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>) -> Vector<S, f32> {
        let a = x.abs();
        let one = 1.0f32.splat::<S>();
        let two = 2.0f32.splat::<S>();
        let half = 0.5f32.splat::<S>();

        // Both ranges are half of `Ln1p` of their own argument.
        let small = two * a + two * a * a / (one - a);
        let large = two * (a / (one - a));
        let y = half * Ln1p.apply(a.lt(half).select(small, large));

        // Below 2^-32 atanh(x) rounds to x.
        let y = a.lt(f32::from_bits(0x2f80_0000).splat()).select(a, y);
        with_sign_of(y, x)
    }
}

/// `tanh(x)` for f32 lanes, on `|x|` with the sign restored: the Cephes
/// `tanhf` polynomial below `|x| = 0.625` and `1 - 2 / (e^(2|x|) + 1)` above,
/// which reaches 1 once the exponential overflows. NaN passes through.
#[derive(Clone, Copy)]
pub struct Tanh;

impl LaneOp<f32> for Tanh {
    #[allow(clippy::excessive_precision)]
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>) -> Vector<S, f32> {
        let a = x.abs();
        let one = 1.0f32.splat::<S>();

        let z = x * x;
        let p = horner(
            z,
            [
                -5.70498872745e-3,
                2.06390887954e-2,
                -5.37397155531e-2,
                1.33314422036e-1,
                -3.33332819422e-1,
            ],
        );
        let small = (p * z).mul_add(a, a);

        let large = one - 2.0f32.splat() / (exp_f32::<S>(a + a) + one);
        let y = a.lt(0.625f32.splat()).select(small, large);
        with_sign_of(y, x)
    }
}

/// `erf(x)` for f32 lanes, on fdlibm's rational approximations. Above
/// `|x| = 1.25` erf is `1 - erfc`, and erfc is below 0.08 there, so its
/// `e^(-x^2)` factor is taken with one [`exp_f32`] instead of fdlibm's
/// split pair: the error it adds is far below an ulp of the result. NaN
/// passes through.
#[allow(clippy::excessive_precision)]
#[inline(always)]
pub fn erf_f32<S: Simd>(x: Vector<S, f32>) -> Vector<S, f32> {
    let a = x.abs();
    let one = 1.0f32.splat::<S>();

    // |x| < 0.84375: |x| + |x| P(x^2) / Q(x^2).
    let z = a * a;
    let p_small = horner(
        z,
        [
            -2.3763017452e-05,
            -5.7702702470e-03,
            -2.8481749818e-02,
            -3.2504209876e-01,
            1.2837916613e-01,
        ],
    );
    let q_small = horner(
        z,
        [
            -3.9602282413e-06,
            1.3249473704e-04,
            5.0813062117e-03,
            6.5022252500e-02,
            3.9791721106e-01,
            1.0,
        ],
    );
    // 0.84375 <= |x| < 1.25: erx + P(s) / Q(s) with s = |x| - 1, where erx is
    // fdlibm's constant 0.84506 (not erf(1); P absorbs the difference).
    let s = a - one;
    let p_mid = horner(
        s,
        [
            -2.1663755178e-03,
            3.5478305072e-02,
            -1.1089469492e-01,
            3.1834661961e-01,
            -3.7220788002e-01,
            4.1485610604e-01,
            -2.3621185683e-03,
        ],
    );
    let q_mid = horner(
        s,
        [
            1.1984500103e-02,
            1.3637083583e-02,
            1.2617121637e-01,
            7.1828655899e-02,
            5.4039794207e-01,
            1.0642088205e-01,
            1.0,
        ],
    );
    // |x| >= 1.25: 1 - e^(-x^2 - 0.5625 + R(t) / S(t)) / |x| with t = 1 / x^2.
    let inv_a = one / a;
    let t = inv_a * inv_a;
    let r_near = horner(
        t,
        [
            -9.8143291473e+00,
            -8.1287437439e+01,
            -1.8460508728e+02,
            -1.6239666748e+02,
            -6.2375331879e+01,
            -1.0558626175e+01,
            -6.9385856390e-01,
            -9.8649440333e-03,
        ],
    );
    let s_near = horner(
        t,
        [
            -6.0424413532e-02,
            6.5702495575e+00,
            1.0863500214e+02,
            4.2900814819e+02,
            6.4538726807e+02,
            4.3456588745e+02,
            1.3765776062e+02,
            1.9651271820e+01,
            1.0,
        ],
    );
    let r_far = horner(
        t,
        [
            -4.8351919556e+02,
            -1.0250950928e+03,
            -6.3756646729e+02,
            -1.6063638306e+02,
            -1.7757955551e+01,
            -7.9928326607e-01,
            -9.8649431020e-03,
        ],
    );
    let s_far = horner(
        t,
        [
            -2.2440952301e+01,
            4.7452853394e+02,
            2.5530502930e+03,
            3.1998581543e+03,
            1.5367296143e+03,
            3.2579251099e+02,
            3.0338060379e+01,
            1.0,
        ],
    );
    // fdlibm's 1 / 0.35 boundary, 0x4036db6d.
    let is_near = || a.lt(f32::from_bits(0x4036_db6d).splat());
    let r_tail = is_near().select(r_near, r_far);
    let s_tail = is_near().select(s_near, s_far);

    // Each lane needs one of the four quotients, so the numerators and
    // denominators are selected first and divided once. `Mask` isn't `Copy`,
    // so each select takes a fresh comparison.
    let is_small = || a.lt(0.84375f32.splat());
    let is_head = || a.lt(1.25f32.splat());
    let num = is_head().select(is_small().select(p_small, p_mid), r_tail);
    let den = is_head().select(is_small().select(q_small, q_mid), s_tail);
    let q = num / den;

    let head = is_small().select(q.mul_add(a, a), q + 8.4506291151e-01f32.splat());
    let tail = one - exp_f32::<S>(q - z - 0.5625f32.splat()) * inv_a;
    let y = is_head().select(head, tail);
    with_sign_of(y, x)
}

/// [`erf_f32`] as a [`LaneOp`].
#[derive(Clone, Copy)]
pub struct Erf;

impl LaneOp<f32> for Erf {
    #[inline(always)]
    fn apply<S: Simd>(self, v: Vector<S, f32>) -> Vector<S, f32> {
        erf_f32::<S>(v)
    }
}

/// The exact GELU, `x / 2 * (1 + erf(x / sqrt(2)))`, for f32 lanes.
#[derive(Clone, Copy)]
pub struct Gelu;

impl LaneOp<f32> for Gelu {
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>) -> Vector<S, f32> {
        x * std_normal_cdf(x)
    }
}

/// The standard normal CDF, `(1 + erf(x / sqrt(2))) / 2`.
#[inline(always)]
fn std_normal_cdf<S: Simd>(x: Vector<S, f32>) -> Vector<S, f32> {
    let erf = erf_f32::<S>(x * core::f32::consts::FRAC_1_SQRT_2.splat());
    0.5f32.splat::<S>() * (erf + 1.0f32.splat())
}

/// The derivative of [`Gelu`], `Phi(x) + x phi(x)` with `Phi` and `phi` the
/// standard normal CDF and density.
#[derive(Clone, Copy)]
pub struct GeluGrad;

impl LaneOp<f32> for GeluGrad {
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>) -> Vector<S, f32> {
        // 1 / sqrt(2 pi)
        const INV_SQRT_2PI: f32 = 0.398_942_3;
        let pdf = INV_SQRT_2PI.splat::<S>() * exp_f32::<S>((-0.5f32).splat::<S>() * x * x);
        x.mul_add(pdf, std_normal_cdf(x))
    }
}

/// Largest |x| the SIMD sin, cos and tan reduce themselves: up to here the
/// quadrant count has at most 12 bits, so each product with the 12-bit
/// pieces of pi/2 in [`reduce_half_pi`] is exact. A vector with a larger lane
/// goes through libm instead.
const TRIG_MAX: f32 = 6433.0;

/// Splits `a` in [0, [`TRIG_MAX`]] into `j * pi/2 + r`, with |r| at most
/// slightly over pi/4 where `a * 2/pi` rounds across a quadrant boundary,
/// returning `r` and `j`. pi/2 is taken in four pieces, the first three with
/// at most 12 significant bits.
#[inline(always)]
fn reduce_half_pi<S: Simd>(a: Vector<S, f32>) -> (Vector<S, f32>, Vector<S, i32>) {
    // Rounds a * 2/pi to an integer as in `exp_f32`.
    let magic = 12_582_912.0f32.splat::<S>();
    let t = a * core::f32::consts::FRAC_2_PI.splat() + magic;
    let jf = t - magic;
    let j = t.reinterpret::<i32>() - magic.reinterpret::<i32>();
    let r = a - jf * f32::from_bits(0x3fc9_0000).splat();
    let r = r - jf * f32::from_bits(0x39fd_a000).splat();
    let r = r - jf * f32::from_bits(0x33a2_2000).splat();
    let r = r - jf * f32::from_bits(0x2c34_611a).splat();
    (r, j)
}

/// `sin(r)` and `cos(r)` for |r| near or below pi/4, on the Cephes `sinf` and `cosf`
/// polynomials.
#[allow(clippy::excessive_precision)]
#[inline(always)]
fn sin_cos_poly<S: Simd>(r: Vector<S, f32>) -> (Vector<S, f32>, Vector<S, f32>) {
    let z = r * r;
    let ps = horner(z, [-1.9515295891e-4, 8.3321608736e-3, -1.6666654611e-1]);
    let sin = (ps * z).mul_add(r, r);
    let pc = horner(
        z,
        [
            2.443315711809948e-5,
            -1.388731625493765e-3,
            4.166664568298827e-2,
        ],
    );
    let cos = (pc * z).mul_add(z, (-0.5f32).splat::<S>().mul_add(z, 1.0f32.splat()));
    (sin, cos)
}

/// `if_odd` in lanes where `j` is odd and `if_even` elsewhere, by bit masks,
/// since `j` is an integer vector.
#[inline(always)]
fn select_odd<S: Simd>(
    j: Vector<S, i32>,
    if_odd: Vector<S, f32>,
    if_even: Vector<S, f32>,
) -> Vector<S, f32> {
    let odd = 0i32.splat::<S>() - (j & 1i32.splat());
    let bits = (if_odd.reinterpret::<i32>() & odd) | (if_even.reinterpret::<i32>() & !odd);
    bits.reinterpret::<f32>()
}

/// Flips the sign of `v` in lanes where `bit` of `j` is set.
#[inline(always)]
fn flip_sign_on_bit<S: Simd>(v: Vector<S, f32>, j: Vector<S, i32>, bit: u32) -> Vector<S, f32> {
    let flip = (j & (1i32 << bit).splat()) << (31 - bit);
    (v.reinterpret::<i32>() ^ flip).reinterpret::<f32>()
}

/// `f` of each lane of `x`, one at a time.
#[inline(never)]
fn per_lane<S: Simd>(x: Vector<S, f32>, f: fn(f32) -> f32) -> Vector<S, f32> {
    let mut buf = [0.0f32; MAX_LANES];
    // SAFETY: a vector has at most `MAX_LANES` f32 lanes.
    unsafe { vstore_unaligned::<S, f32>(buf.as_mut_ptr(), x) };
    for v in &mut buf[..f32::lanes::<S>()] {
        *v = f(*v);
    }
    // SAFETY: as above.
    unsafe { vload_unaligned::<S, f32>(buf.as_ptr()) }
}

/// `sin(x)` for f32 lanes, on |x| with the sign restored. A vector with a lane
/// beyond [`TRIG_MAX`] (inf included) goes through `libm::sinf`.
#[derive(Clone, Copy)]
pub struct Sin;

impl LaneOp<f32> for Sin {
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>) -> Vector<S, f32> {
        let a = x.abs();
        if a.gt(TRIG_MAX.splat()).any() {
            return per_lane(x, libm::sinf);
        }
        let (r, j) = reduce_half_pi(a);
        let (sin, cos) = sin_cos_poly(r);
        // Quadrants 1 and 3 take cos, and 2 and 3 negate.
        let y = flip_sign_on_bit(select_odd(j, cos, sin), j, 1);
        xor_sign_of(y, x)
    }
}

/// `cos(x)` for f32 lanes, on |x|. A vector with a lane beyond [`TRIG_MAX`]
/// (inf included) goes through `libm::cosf`.
#[derive(Clone, Copy)]
pub struct Cos;

impl LaneOp<f32> for Cos {
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>) -> Vector<S, f32> {
        let a = x.abs();
        if a.gt(TRIG_MAX.splat()).any() {
            return per_lane(x, libm::cosf);
        }
        let (r, j) = reduce_half_pi(a);
        let (sin, cos) = sin_cos_poly(r);
        // cos(x) = sin(x + pi/2): one quadrant on from `Sin`.
        let j = j + 1i32.splat();
        flip_sign_on_bit(select_odd(j, cos, sin), j, 1)
    }
}

/// `tan(x)` for f32 lanes, on |x| with the sign restored: `sin(r) / cos(r)` in
/// even quadrants and `-cos(r) / sin(r)` in odd ones. A vector with a lane
/// beyond [`TRIG_MAX`] (inf included) goes through `libm::tanf`.
#[derive(Clone, Copy)]
pub struct Tan;

impl LaneOp<f32> for Tan {
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>) -> Vector<S, f32> {
        let a = x.abs();
        if a.gt(TRIG_MAX.splat()).any() {
            return per_lane(x, libm::tanf);
        }
        let (r, j) = reduce_half_pi(a);
        let (sin, cos) = sin_cos_poly(r);
        let y = select_odd(j, cos, sin) / select_odd(j, sin, cos);
        xor_sign_of(flip_sign_on_bit(y, j, 0), x)
    }
}

/// `asin(w)` for |w| <= 0.5, on the Cephes `asinf` polynomial.
#[allow(clippy::excessive_precision)]
#[inline(always)]
fn asin_poly<S: Simd>(w: Vector<S, f32>, z: Vector<S, f32>) -> Vector<S, f32> {
    let p = horner(
        z,
        [
            4.2163199048e-2,
            2.4181311049e-2,
            4.5470025998e-2,
            7.4953002686e-2,
            1.6666752422e-1,
        ],
    );
    (p * z).mul_add(w, w)
}

/// `atan(x)` for f32 lanes, on |x| with the sign restored: Cephes `atanf`,
/// which takes `-1/|x|` above tan(3pi/8) and `(|x| - 1) / (|x| + 1)` above
/// tan(pi/8), adding pi/2 or pi/4. The three reductions share one division.
#[derive(Clone, Copy)]
pub struct Atan;

impl LaneOp<f32> for Atan {
    #[allow(clippy::excessive_precision)]
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>) -> Vector<S, f32> {
        let a = x.abs();
        let one = 1.0f32.splat::<S>();
        // `Mask` isn't `Copy`, so each select takes a fresh comparison.
        let big = || a.gt(2.414_213_562_373_095f32.splat());
        let mid = || a.gt(0.414_213_562_373_095f32.splat());

        let num = big().select((-1.0f32).splat(), mid().select(a - one, a));
        let den = big().select(a, mid().select(a + one, one));
        let r = num / den;
        let offset = big().select(
            core::f32::consts::FRAC_PI_2.splat(),
            mid().select(core::f32::consts::FRAC_PI_4.splat(), 0.0f32.splat()),
        );

        let z = r * r;
        let p = horner(
            z,
            [
                8.05374449538e-2,
                -1.38776856032e-1,
                1.99777106478e-1,
                -3.33329491539e-1,
            ],
        );
        with_sign_of(offset + (p * z).mul_add(r, r), x)
    }
}

/// `asin(x)` for f32 lanes, on |x| with the sign restored: Cephes `asinf`,
/// which above |x| = 0.5 takes `pi/2 - 2 asin(sqrt((1 - |x|) / 2))`. Inputs
/// beyond +-1 give NaN.
#[derive(Clone, Copy)]
pub struct Asin;

impl LaneOp<f32> for Asin {
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>) -> Vector<S, f32> {
        let a = x.abs();
        let half = 0.5f32.splat::<S>();
        let high = || a.gt(half);

        let z = high().select(half * (1.0f32.splat::<S>() - a), a * a);
        let w = high().select(z.sqrt(), a);
        let p = asin_poly(w, z);
        let y = high().select(core::f32::consts::FRAC_PI_2.splat::<S>() - (p + p), p);
        with_sign_of(y, x)
    }
}

/// `acos(x)` for f32 lanes: Cephes `acosf`, as `pi - 2 asin(sqrt((1 + x) / 2))`
/// below -0.5, `2 asin(sqrt((1 - x) / 2))` above 0.5 and `pi/2 - asin(x)`
/// between, all on one `asin` of an argument within +-0.5. Inputs beyond +-1
/// give NaN.
#[derive(Clone, Copy)]
pub struct Acos;

impl LaneOp<f32> for Acos {
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>) -> Vector<S, f32> {
        let one = 1.0f32.splat::<S>();
        let half = 0.5f32.splat::<S>();
        let low = || x.lt((-0.5f32).splat());
        let high = || x.gt(half);

        let w = low().select(
            (half * (one + x)).sqrt(),
            high().select((half * (one - x)).sqrt(), x),
        );
        let s = asin_poly(w, w * w);
        low().select(
            core::f32::consts::PI.splat::<S>() - (s + s),
            high().select(s + s, core::f32::consts::FRAC_PI_2.splat::<S>() - s),
        )
    }
}

/// `atan2(y, x)` for f32 lanes, with `y` first: `atan(|y| / |x|)` reflected
/// to `pi - t` where `x` has its sign bit set, then given the sign of `y`. The
/// IEEE special cases come out as in libm: both zeros give +-0 or +-pi by the
/// sign of `x`, both infinities give +-pi/4 or +-3pi/4, and NaN gives NaN.
#[derive(Clone, Copy)]
pub struct Atan2;

impl LaneOp2<f32> for Atan2 {
    #[allow(clippy::excessive_precision)]
    #[inline(always)]
    fn apply<S: Simd>(self, y: Vector<S, f32>, x: Vector<S, f32>) -> Vector<S, f32> {
        let a = y.abs();
        let b = x.abs();
        let inf = f32::INFINITY.splat::<S>();

        let t = Atan.apply(a / b);
        // 0/0 and 0/inf are 0 and inf/inf is pi/4 in the first quadrant.
        let t = a.eq(0.0f32.splat()).select(0.0f32.splat(), t);
        let t = (a.eq(inf) & b.eq(inf)).select(core::f32::consts::FRAC_PI_4.splat(), t);
        // pi - t with pi in two parts, the low one -8.742278e-8.
        let reflected = ((-8.742_278e-8f32).splat::<S>() - t) + core::f32::consts::PI.splat();
        let x_negative = x.reinterpret::<i32>().lt(0i32.splat());
        let t = select_by_int_mask(x_negative, reflected, t);

        let out = with_sign_of(t, y);
        // Either NaN gives NaN, which `a == 0` above could have hidden.
        (!(x.eq(x) & y.eq(y))).select(x + y, out)
    }
}

/// `if_true` where the lanes of the i32 comparison `mask` hold, else
/// `if_false`.
#[inline(always)]
fn select_by_int_mask<S: Simd>(
    mask: Mask<S, i32>,
    if_true: Vector<S, f32>,
    if_false: Vector<S, f32>,
) -> Vector<S, f32> {
    mask.select(if_true.reinterpret::<i32>(), if_false.reinterpret::<i32>())
        .reinterpret::<f32>()
}

/// `hypot(x, y)` for f32 lanes, as `sqrt(x^2 + y^2)` after scaling both by a
/// power of two from the larger exponent, so the larger square neither
/// overflows nor underflows and scaling it is exact. An infinite input gives +inf, even
/// against NaN, as IEEE specifies.
#[derive(Clone, Copy)]
pub struct Hypot;

impl LaneOp2<f32> for Hypot {
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>, y: Vector<S, f32>) -> Vector<S, f32> {
        let a = x.abs();
        let b = y.abs();
        let inf = f32::INFINITY.splat::<S>();

        // The larger magnitude's biased exponent `e` picks the scale 2^(127 - e),
        // which brings it near 1. Its biased exponent 254 - e is kept at 1 or
        // more so it stays a normal number; the inputs in the top binade then
        // land in [2, 4) instead, still far from overflow.
        let larger = a.ge(b).select(a, b);
        let e = larger.reinterpret::<i32>() >> 23;
        let scale_exp = 254i32.splat::<S>() - e;
        let scale_exp = scale_exp.lt(1i32.splat()).select(1i32.splat(), scale_exp);
        let scale = (scale_exp << 23).reinterpret::<f32>();

        let sa = a * scale;
        let sb = b * scale;
        let h = (sa * sa + sb * sb).sqrt() / scale;
        (a.eq(inf) | b.eq(inf)).select(inf, h)
    }
}

/// The GELU backward pass, [`GeluGrad`] of `x` times the incoming gradient,
/// in one pass.
#[derive(Clone, Copy)]
pub struct GeluBackward;

impl LaneOp2<f32> for GeluBackward {
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>, grad: Vector<S, f32>) -> Vector<S, f32> {
        GeluGrad.apply(x) * grad
    }
}

/// The log-sigmoid backward pass, [`LogSigmoidGrad`] of `x` times the
/// incoming gradient, in one pass.
#[derive(Clone, Copy)]
pub struct LogSigmoidBackward;

impl LaneOp2<f32> for LogSigmoidBackward {
    #[inline(always)]
    fn apply<S: Simd>(self, x: Vector<S, f32>, grad: Vector<S, f32>) -> Vector<S, f32> {
        LogSigmoidGrad.apply(x) * grad
    }
}

/// `v` with its sign flipped where `sign` is negative.
#[inline(always)]
fn xor_sign_of<S: Simd>(v: Vector<S, f32>, sign: Vector<S, f32>) -> Vector<S, f32> {
    let sign_bit = sign.reinterpret::<i32>() & i32::MIN.splat();
    (v.reinterpret::<i32>() ^ sign_bit).reinterpret::<f32>()
}

// ============================================================================
// f32 comparison ops
// ============================================================================

#[derive(Clone, Copy)]
pub enum CmpOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

#[inline]
pub fn cmp_f32(a: &[f32], b: &[f32], out: &mut [u8], op: CmpOp) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        cmp_f32_par(a, b, out, op);
        return;
    }

    cmp_f32_seq(a, b, out, op);
}

/// Comparison kernel using simple loops that LLVM autovectorizes.
///
/// Autovectorization outperforms explicit SIMD here because comparisons
/// produce u8 output from f32 input (4:1 size ratio). LLVM batches 16+
/// comparisons and packs results into a single wide vector store, while
/// explicit SIMD (store_as_bool) writes only `lanes` bytes per iteration.
#[inline]
fn cmp_f32_seq(a: &[f32], b: &[f32], out: &mut [u8], op: CmpOp) {
    match op {
        CmpOp::Gt => {
            for ((a, b), o) in a.iter().zip(b).zip(out.iter_mut()) {
                *o = (*a > *b) as u8;
            }
        }
        CmpOp::Ge => {
            for ((a, b), o) in a.iter().zip(b).zip(out.iter_mut()) {
                *o = (*a >= *b) as u8;
            }
        }
        CmpOp::Lt => {
            for ((a, b), o) in a.iter().zip(b).zip(out.iter_mut()) {
                *o = (*a < *b) as u8;
            }
        }
        CmpOp::Le => {
            for ((a, b), o) in a.iter().zip(b).zip(out.iter_mut()) {
                *o = (*a <= *b) as u8;
            }
        }
        CmpOp::Eq => {
            for ((a, b), o) in a.iter().zip(b).zip(out.iter_mut()) {
                *o = (*a == *b) as u8;
            }
        }
        CmpOp::Ne => {
            for ((a, b), o) in a.iter().zip(b).zip(out.iter_mut()) {
                *o = (*a != *b) as u8;
            }
        }
    }
}

#[cfg(feature = "rayon")]
fn cmp_f32_par(a: &[f32], b: &[f32], out: &mut [u8], op: CmpOp) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(a.len());
            cmp_f32_seq(&a[start..end], &b[start..end], out_chunk, op);
        });
}

#[inline]
pub fn cmp_scalar_f32(a: &[f32], scalar: f32, out: &mut [u8], op: CmpOp) {
    debug_assert_eq!(a.len(), out.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        cmp_scalar_f32_par(a, scalar, out, op);
        return;
    }

    cmp_scalar_f32_seq(a, scalar, out, op);
}

/// Scalar comparison kernel using simple loops that LLVM autovectorizes.
/// See `cmp_f32_seq` for rationale.
#[inline]
fn cmp_scalar_f32_seq(a: &[f32], scalar: f32, out: &mut [u8], op: CmpOp) {
    match op {
        CmpOp::Gt => {
            for (a, o) in a.iter().zip(out.iter_mut()) {
                *o = (*a > scalar) as u8;
            }
        }
        CmpOp::Ge => {
            for (a, o) in a.iter().zip(out.iter_mut()) {
                *o = (*a >= scalar) as u8;
            }
        }
        CmpOp::Lt => {
            for (a, o) in a.iter().zip(out.iter_mut()) {
                *o = (*a < scalar) as u8;
            }
        }
        CmpOp::Le => {
            for (a, o) in a.iter().zip(out.iter_mut()) {
                *o = (*a <= scalar) as u8;
            }
        }
        CmpOp::Eq => {
            for (a, o) in a.iter().zip(out.iter_mut()) {
                *o = (*a == scalar) as u8;
            }
        }
        CmpOp::Ne => {
            for (a, o) in a.iter().zip(out.iter_mut()) {
                *o = (*a != scalar) as u8;
            }
        }
    }
}

#[cfg(feature = "rayon")]
fn cmp_scalar_f32_par(a: &[f32], scalar: f32, out: &mut [u8], op: CmpOp) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(a.len());
            cmp_scalar_f32_seq(&a[start..end], scalar, out_chunk, op);
        });
}

// ============================================================================
// u8 boolean ops
// ============================================================================

macro_rules! define_bool_binary_u8_op {
    ($pub_fn:ident, $seq_fn:ident, $par_fn:ident,
     $inplace_pub:ident, $inplace_seq:ident, $inplace_par:ident,
     $trait:ident, $method:ident, $op:tt) => {
        #[inline]
        pub fn $pub_fn(a: &[u8], b: &[u8], out: &mut [u8]) {
            debug_assert_eq!(a.len(), b.len());
            debug_assert_eq!(a.len(), out.len());

            #[cfg(feature = "rayon")]
            if a.len() >= PARALLEL_THRESHOLD {
                $par_fn(a, b, out);
                return;
            }

            $seq_fn(a, b, out);
        }

        #[macerator::with_simd]
        fn $seq_fn<S: Simd>(a: &[u8], b: &[u8], out: &mut [u8]) {
            let lanes = S::lanes8();
            let len = a.len();
            let simd_len = len / lanes * lanes;

            let mut i = 0;
            while i < simd_len {
                unsafe {
                    let va = vload_unaligned::<S, u8>(a.as_ptr().add(i));
                    let vb = vload_unaligned::<S, u8>(b.as_ptr().add(i));
                    vstore_unaligned::<S, u8>(
                        out.as_mut_ptr().add(i),
                        <u8 as $trait>::$method::<S>(va, vb),
                    );
                }
                i += lanes;
            }

            for j in simd_len..len {
                out[j] = a[j] $op b[j];
            }
        }

        #[cfg(feature = "rayon")]
        fn $par_fn(a: &[u8], b: &[u8], out: &mut [u8]) {
            out.par_chunks_mut(CHUNK_SIZE)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let start = chunk_idx * CHUNK_SIZE;
                    let end = (start + CHUNK_SIZE).min(a.len());
                    $seq_fn(&a[start..end], &b[start..end], out_chunk);
                });
        }

        #[inline]
        pub fn $inplace_pub(a: &mut [u8], b: &[u8]) {
            debug_assert_eq!(a.len(), b.len());

            #[cfg(feature = "rayon")]
            if a.len() >= PARALLEL_THRESHOLD {
                $inplace_par(a, b);
                return;
            }

            $inplace_seq(a, b);
        }

        #[allow(clippy::assign_op_pattern)]
        #[macerator::with_simd]
        fn $inplace_seq<S: Simd>(a: &mut [u8], b: &[u8]) {
            let lanes = S::lanes8();
            let len = a.len();
            let simd_len = len / lanes * lanes;

            let mut i = 0;
            while i < simd_len {
                unsafe {
                    let va = vload_unaligned::<S, u8>(a.as_ptr().add(i));
                    let vb = vload_unaligned::<S, u8>(b.as_ptr().add(i));
                    vstore_unaligned::<S, u8>(
                        a.as_mut_ptr().add(i),
                        <u8 as $trait>::$method::<S>(va, vb),
                    );
                }
                i += lanes;
            }

            for j in simd_len..len {
                a[j] = a[j] $op b[j];
            }
        }

        #[cfg(feature = "rayon")]
        fn $inplace_par(a: &mut [u8], b: &[u8]) {
            a.par_chunks_mut(CHUNK_SIZE)
                .zip(b.par_chunks(CHUNK_SIZE))
                .for_each(|(a_chunk, b_chunk)| {
                    $inplace_seq(a_chunk, b_chunk);
                });
        }
    };
}

define_bool_binary_u8_op!(
    bool_and_u8, bool_and_u8_seq, bool_and_u8_par,
    bool_and_inplace_u8, bool_and_inplace_u8_seq, bool_and_inplace_u8_par,
    VBitAnd, vbitand, &);
define_bool_binary_u8_op!(
    bool_or_u8, bool_or_u8_seq, bool_or_u8_par,
    bool_or_inplace_u8, bool_or_inplace_u8_seq, bool_or_inplace_u8_par,
    VBitOr, vbitor, |);
define_bool_binary_u8_op!(
    bool_xor_u8, bool_xor_u8_seq, bool_xor_u8_par,
    bool_xor_inplace_u8, bool_xor_inplace_u8_seq, bool_xor_inplace_u8_par,
    VBitXor, vbitxor, ^);

// Boolean NOT is special (unary), implemented separately.

#[inline]
pub fn bool_not_u8(a: &[u8], out: &mut [u8]) {
    debug_assert_eq!(a.len(), out.len());

    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        bool_not_u8_par(a, out);
        return;
    }

    bool_not_u8_seq(a, out);
}

#[macerator::with_simd]
fn bool_not_u8_seq<S: Simd>(a: &[u8], out: &mut [u8]) {
    let lanes = S::lanes8();
    let len = a.len();
    let simd_len = len / lanes * lanes;

    let zeros = 0u8.splat::<S>();

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let va = vload_unaligned::<S, u8>(a.as_ptr().add(i));
            let mask = va.eq(zeros);
            mask.store_as_bool(out.as_mut_ptr().add(i) as *mut bool);
        }
        i += lanes;
    }

    for j in simd_len..len {
        out[j] = (a[j] == 0) as u8;
    }
}

#[cfg(feature = "rayon")]
fn bool_not_u8_par(a: &[u8], out: &mut [u8]) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(a.len());
            bool_not_u8_seq(&a[start..end], out_chunk);
        });
}

#[inline]
pub fn bool_not_inplace_u8(a: &mut [u8]) {
    #[cfg(feature = "rayon")]
    if a.len() >= PARALLEL_THRESHOLD {
        bool_not_inplace_u8_par(a);
        return;
    }

    bool_not_inplace_u8_seq(a);
}

#[macerator::with_simd]
fn bool_not_inplace_u8_seq<S: Simd>(a: &mut [u8]) {
    let lanes = S::lanes8();
    let len = a.len();
    let simd_len = len / lanes * lanes;

    let zeros = 0u8.splat::<S>();

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let va = vload_unaligned::<S, u8>(a.as_ptr().add(i));
            let mask = va.eq(zeros);
            mask.store_as_bool(a.as_mut_ptr().add(i) as *mut bool);
        }
        i += lanes;
    }

    for v in &mut a[simd_len..len] {
        *v = (*v == 0) as u8;
    }
}

#[cfg(feature = "rayon")]
fn bool_not_inplace_u8_par(a: &mut [u8]) {
    a.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
        bool_not_inplace_u8_seq(chunk);
    });
}

// ============================================================================
// Mask select (mask_where / mask_fill) via bitwise blend
// ============================================================================

/// Conditional select: `out[i] = if mask[i] != 0 { value[i] } else { tensor[i] }`
///
/// Operates on f32 data reinterpreted as u32 for bitwise blend.
/// Uses SIMD bitwise ops: `(mask & value) | (!mask & tensor)`.
#[inline]
pub fn mask_where_f32(tensor: &[f32], mask: &[u8], value: &[f32], out: &mut [f32]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), value.len());
    debug_assert_eq!(tensor.len(), out.len());

    let t = bytemuck::cast_slice::<f32, u32>(tensor);
    let v = bytemuck::cast_slice::<f32, u32>(value);
    let o = bytemuck::cast_slice_mut::<f32, u32>(out);

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_where_u32_par(t, mask, v, o);
        return;
    }

    mask_blend_u32(t, mask, v, o);
}

/// Conditional select for f64.
#[inline]
pub fn mask_where_f64(tensor: &[f64], mask: &[u8], value: &[f64], out: &mut [f64]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), value.len());
    debug_assert_eq!(tensor.len(), out.len());

    let t = bytemuck::cast_slice::<f64, u64>(tensor);
    let v = bytemuck::cast_slice::<f64, u64>(value);
    let o = bytemuck::cast_slice_mut::<f64, u64>(out);

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_where_u64_par(t, mask, v, o);
        return;
    }

    mask_blend_u64(t, mask, v, o);
}

/// Conditional select for i64.
#[inline]
pub fn mask_where_i64(tensor: &[i64], mask: &[u8], value: &[i64], out: &mut [i64]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), value.len());
    debug_assert_eq!(tensor.len(), out.len());

    let t = bytemuck::cast_slice::<i64, u64>(tensor);
    let v = bytemuck::cast_slice::<i64, u64>(value);
    let o = bytemuck::cast_slice_mut::<i64, u64>(out);

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_where_u64_par(t, mask, v, o);
        return;
    }

    mask_blend_u64(t, mask, v, o);
}

/// Conditional select for u8 (bool tensors).
#[inline]
pub fn mask_where_u8(tensor: &[u8], mask: &[u8], value: &[u8], out: &mut [u8]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), value.len());
    debug_assert_eq!(tensor.len(), out.len());

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_where_u8_par(tensor, mask, value, out);
        return;
    }

    mask_where_u8_seq(tensor, mask, value, out);
}

/// Conditional fill: `out[i] = if mask[i] != 0 { fill_value } else { tensor[i] }`
#[inline]
pub fn mask_fill_f32(tensor: &[f32], mask: &[u8], fill_value: f32, out: &mut [f32]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), out.len());

    let t = bytemuck::cast_slice::<f32, u32>(tensor);
    let o = bytemuck::cast_slice_mut::<f32, u32>(out);
    let fill_bits = fill_value.to_bits();

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_fill_u32_par(t, mask, fill_bits, o);
        return;
    }

    mask_blend_fill_u32(t, mask, fill_bits, o);
}

/// Conditional fill for f64.
#[inline]
pub fn mask_fill_f64(tensor: &[f64], mask: &[u8], fill_value: f64, out: &mut [f64]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), out.len());

    let t = bytemuck::cast_slice::<f64, u64>(tensor);
    let o = bytemuck::cast_slice_mut::<f64, u64>(out);
    let fill_bits = fill_value.to_bits();

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_fill_u64_par(t, mask, fill_bits, o);
        return;
    }

    mask_blend_fill_u64(t, mask, fill_bits, o);
}

/// Conditional fill for i64.
#[inline]
pub fn mask_fill_i64(tensor: &[i64], mask: &[u8], fill_value: i64, out: &mut [i64]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), out.len());

    let t = bytemuck::cast_slice::<i64, u64>(tensor);
    let o = bytemuck::cast_slice_mut::<i64, u64>(out);
    let fill_bits = fill_value as u64;

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_fill_u64_par(t, mask, fill_bits, o);
        return;
    }

    mask_blend_fill_u64(t, mask, fill_bits, o);
}

/// Conditional fill for u8 (bool tensors).
#[inline]
pub fn mask_fill_u8(tensor: &[u8], mask: &[u8], fill_value: u8, out: &mut [u8]) {
    debug_assert_eq!(tensor.len(), mask.len());
    debug_assert_eq!(tensor.len(), out.len());

    #[cfg(feature = "rayon")]
    if tensor.len() >= PARALLEL_THRESHOLD {
        mask_fill_u8_par(tensor, mask, fill_value, out);
        return;
    }

    mask_fill_u8_seq(tensor, mask, fill_value, out);
}

// -- Branchless bitwise blend kernels --
//
// These use a tight branchless loop that LLVM autovectorizes with native
// u8->u32/u64 widening instructions (NEON ushll, AVX2 vpmovzxbd).
//
// Mask values must be exactly 0 or 1 (Burn bool tensor invariant).
// wrapping_sub(0, 0)=0x00, wrapping_sub(0, 1)=0xFF..FF. Other values
// produce partial masks and corrupt output.

#[inline]
fn mask_blend_u32(tensor: &[u32], mask: &[u8], value: &[u32], out: &mut [u32]) {
    for i in 0..tensor.len() {
        let m = 0u32.wrapping_sub(mask[i] as u32);
        out[i] = (value[i] & m) | (tensor[i] & !m);
    }
}

#[inline]
fn mask_blend_fill_u32(tensor: &[u32], mask: &[u8], fill_bits: u32, out: &mut [u32]) {
    for i in 0..tensor.len() {
        let m = 0u32.wrapping_sub(mask[i] as u32);
        out[i] = (fill_bits & m) | (tensor[i] & !m);
    }
}

#[inline]
fn mask_blend_u64(tensor: &[u64], mask: &[u8], value: &[u64], out: &mut [u64]) {
    for i in 0..tensor.len() {
        let m = 0u64.wrapping_sub(mask[i] as u64);
        out[i] = (value[i] & m) | (tensor[i] & !m);
    }
}

#[inline]
fn mask_blend_fill_u64(tensor: &[u64], mask: &[u8], fill_bits: u64, out: &mut [u64]) {
    for i in 0..tensor.len() {
        let m = 0u64.wrapping_sub(mask[i] as u64);
        out[i] = (fill_bits & m) | (tensor[i] & !m);
    }
}

#[cfg(feature = "rayon")]
fn mask_where_u32_par(tensor: &[u32], mask: &[u8], value: &[u32], out: &mut [u32]) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(tensor.len());
            mask_blend_u32(
                &tensor[start..end],
                &mask[start..end],
                &value[start..end],
                out_chunk,
            );
        });
}

#[cfg(feature = "rayon")]
fn mask_fill_u32_par(tensor: &[u32], mask: &[u8], fill_bits: u32, out: &mut [u32]) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(tensor.len());
            mask_blend_fill_u32(&tensor[start..end], &mask[start..end], fill_bits, out_chunk);
        });
}

#[cfg(feature = "rayon")]
fn mask_where_u64_par(tensor: &[u64], mask: &[u8], value: &[u64], out: &mut [u64]) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(tensor.len());
            mask_blend_u64(
                &tensor[start..end],
                &mask[start..end],
                &value[start..end],
                out_chunk,
            );
        });
}

#[cfg(feature = "rayon")]
fn mask_fill_u64_par(tensor: &[u64], mask: &[u8], fill_bits: u64, out: &mut [u64]) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(tensor.len());
            mask_blend_fill_u64(&tensor[start..end], &mask[start..end], fill_bits, out_chunk);
        });
}

// -- u8 SIMD kernels (for bool tensors) --

#[macerator::with_simd]
fn mask_where_u8_seq<S: Simd>(tensor: &[u8], mask: &[u8], value: &[u8], out: &mut [u8]) {
    let lanes = S::lanes8();
    let len = tensor.len();
    let simd_len = len / lanes * lanes;

    // SIMD subtract wraps: 0-0=0x00, 0-1=0xFF
    let zeros = 0u8.splat::<S>();

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let vm_raw = vload_unaligned::<S, u8>(mask.as_ptr().add(i));
            let vm = zeros - vm_raw; // 0->0x00, 1->0xFF
            let vt = vload_unaligned::<S, u8>(tensor.as_ptr().add(i));
            let vv = vload_unaligned::<S, u8>(value.as_ptr().add(i));
            let selected = (vm & vv) | (!vm & vt);
            vstore_unaligned::<S, u8>(out.as_mut_ptr().add(i), selected);
        }
        i += lanes;
    }

    for j in simd_len..len {
        let m = 0u8.wrapping_sub(mask[j]);
        out[j] = (m & value[j]) | (!m & tensor[j]);
    }
}

#[cfg(feature = "rayon")]
fn mask_where_u8_par(tensor: &[u8], mask: &[u8], value: &[u8], out: &mut [u8]) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(tensor.len());
            mask_where_u8_seq(
                &tensor[start..end],
                &mask[start..end],
                &value[start..end],
                out_chunk,
            );
        });
}

#[macerator::with_simd]
fn mask_fill_u8_seq<S: Simd>(tensor: &[u8], mask: &[u8], fill_value: u8, out: &mut [u8]) {
    let lanes = S::lanes8();
    let len = tensor.len();
    let simd_len = len / lanes * lanes;
    let vfill = fill_value.splat::<S>();
    let zeros = 0u8.splat::<S>();

    let mut i = 0;
    while i < simd_len {
        unsafe {
            let vm_raw = vload_unaligned::<S, u8>(mask.as_ptr().add(i));
            let vm = zeros - vm_raw; // 0->0x00, 1->0xFF
            let vt = vload_unaligned::<S, u8>(tensor.as_ptr().add(i));
            let selected = (vm & vfill) | (!vm & vt);
            vstore_unaligned::<S, u8>(out.as_mut_ptr().add(i), selected);
        }
        i += lanes;
    }

    for j in simd_len..len {
        let m = 0u8.wrapping_sub(mask[j]);
        out[j] = (m & fill_value) | (!m & tensor[j]);
    }
}

#[cfg(feature = "rayon")]
fn mask_fill_u8_par(tensor: &[u8], mask: &[u8], fill_value: u8, out: &mut [u8]) {
    out.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(tensor.len());
            mask_fill_u8_seq(
                &tensor[start..end],
                &mask[start..end],
                fill_value,
                out_chunk,
            );
        });
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_inplace_f32() {
        let mut a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        add_inplace_f32(&mut a, &b);
        assert_eq!(a, [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0]);
    }

    #[test]
    fn test_sub_inplace_f32() {
        let mut a = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        sub_inplace_f32(&mut a, &b);
        assert_eq!(a, [9.0, 18.0, 27.0, 36.0, 45.0]);
    }

    #[test]
    fn test_mul_inplace_f32() {
        let mut a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0f32, 2.0, 2.0, 2.0, 2.0];
        mul_inplace_f32(&mut a, &b);
        assert_eq!(a, [2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_div_inplace_f32() {
        let mut a = [10.0f32, 20.0, 30.0, 40.0];
        let b = [2.0f32, 4.0, 5.0, 8.0];
        div_inplace_f32(&mut a, &b);
        assert_eq!(a, [5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn test_rsub_inplace_f32() {
        // dst = src - dst, i.e. a[i] = b[i] - a[i].
        let mut a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        rsub_inplace_f32(&mut a, &b);
        assert_eq!(a, [9.0, 18.0, 27.0, 36.0, 45.0]);
    }

    #[test]
    fn test_rsub_inplace_f32_long() {
        // Not a multiple of any lane width, so both the SIMD body and
        // the scalar tail run on 4-, 8- and 16-lane targets alike
        // (101 = 25*4 + 1 = 12*8 + 5 = 6*16 + 5).
        let mut a: Vec<f32> = (0..101).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..101).map(|i| (3 * i) as f32).collect();
        rsub_inplace_f32(&mut a, &b);
        let expected: Vec<f32> = (0..101).map(|i| (2 * i) as f32).collect();
        assert_eq!(a, expected);
    }

    #[test]
    fn test_rdiv_inplace_f32() {
        // dst = src / dst, i.e. a[i] = b[i] / a[i]. Length 5 so the
        // scalar tail runs even on a four-lane target.
        let mut a = [2.0f32, 4.0, 5.0, 8.0, 10.0];
        let b = [10.0f32, 20.0, 30.0, 40.0, 70.0];
        rdiv_inplace_f32(&mut a, &b);
        assert_eq!(a, [5.0, 5.0, 6.0, 5.0, 7.0]);
    }

    #[test]
    fn test_rdiv_inplace_f32_long() {
        // 101 for the same reason as the rsub case above; the result is
        // deliberately not the input, so a no-op kernel can't pass.
        let mut a: Vec<f32> = (1..=101).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=101).map(|i| (2 * i * i) as f32).collect();
        rdiv_inplace_f32(&mut a, &b);
        let expected: Vec<f32> = (1..=101).map(|i| (2 * i) as f32).collect();
        assert_eq!(a, expected);
    }

    /// The reversed kernels must agree with the natural kernels run on
    /// swapped inputs: `rsub(a, b)` == `sub(b, a)` element-wise.
    #[test]
    fn test_reversed_kernels_match_swapped_natural_kernels() {
        let x: Vec<f32> = (1..=50).map(|i| i as f32 * 0.75).collect();
        let y: Vec<f32> = (1..=50).map(|i| (51 - i) as f32 * 1.25).collect();

        let mut rsub_out = x.clone();
        rsub_inplace_f32(&mut rsub_out, &y);
        let mut sub_out = y.clone();
        sub_inplace_f32(&mut sub_out, &x);
        assert_eq!(rsub_out, sub_out);

        let mut rdiv_out = x.clone();
        rdiv_inplace_f32(&mut rdiv_out, &y);
        let mut div_out = y.clone();
        div_inplace_f32(&mut div_out, &x);
        assert_eq!(rdiv_out, div_out);
    }

    #[test]
    fn test_cmp_gt_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [2.0f32, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0];
        let mut out = [0u8; 7];
        cmp_f32(&a, &b, &mut out, CmpOp::Gt);
        assert_eq!(out, [0, 0, 1, 0, 1, 1, 1]);
    }

    #[test]
    fn test_cmp_ge_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [2.0f32, 2.0, 2.0, 5.0];
        let mut out = [0u8; 4];
        cmp_f32(&a, &b, &mut out, CmpOp::Ge);
        assert_eq!(out, [0, 1, 1, 0]);
    }

    #[test]
    fn test_cmp_eq_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0f32, 3.0, 3.0, 5.0, 5.0];
        let mut out = [0u8; 5];
        cmp_f32(&a, &b, &mut out, CmpOp::Eq);
        assert_eq!(out, [1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_cmp_ne_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0f32, 3.0, 3.0, 5.0, 5.0];
        let mut out = [0u8; 5];
        cmp_f32(&a, &b, &mut out, CmpOp::Ne);
        assert_eq!(out, [0, 1, 0, 1, 0]);
    }

    #[test]
    fn test_cmp_scalar_gt_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0u8; 5];
        cmp_scalar_f32(&a, 3.0, &mut out, CmpOp::Gt);
        assert_eq!(out, [0, 0, 0, 1, 1]);
    }

    #[test]
    fn test_bool_not_u8() {
        let a = [1u8, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0];
        let mut out = [0u8; 18];
        bool_not_u8(&a, &mut out);
        let expected = [0u8, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_bool_not_inplace_u8() {
        let mut a = [1u8, 0, 1, 0];
        bool_not_inplace_u8(&mut a);
        assert_eq!(a, [0, 1, 0, 1]);
    }

    #[test]
    fn test_bool_and_u8() {
        let a = [1u8, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0];
        let b = [1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1];
        let mut out = [0u8; 18];
        bool_and_u8(&a, &b, &mut out);
        let expected = [1u8, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_bool_or_u8() {
        let a = [1u8, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0];
        let b = [1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1];
        let mut out = [0u8; 18];
        bool_or_u8(&a, &b, &mut out);
        let expected = [1u8, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_bool_xor_u8() {
        let a = [1u8, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0];
        let b = [1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1];
        let mut out = [0u8; 18];
        bool_xor_u8(&a, &b, &mut out);
        let expected = [0u8, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_bool_and_inplace_u8() {
        let mut a = [1u8, 1, 0, 0];
        let b = [1u8, 0, 1, 0];
        bool_and_inplace_u8(&mut a, &b);
        assert_eq!(a, [1, 0, 0, 0]);
    }

    #[test]
    fn test_bool_or_inplace_u8() {
        let mut a = [1u8, 1, 0, 0];
        let b = [1u8, 0, 1, 0];
        bool_or_inplace_u8(&mut a, &b);
        assert_eq!(a, [1, 1, 1, 0]);
    }

    #[test]
    fn test_bool_xor_inplace_u8() {
        let mut a = [1u8, 1, 0, 0];
        let b = [1u8, 0, 1, 0];
        bool_xor_inplace_u8(&mut a, &b);
        assert_eq!(a, [0, 1, 1, 0]);
    }

    #[test]
    fn test_abs_inplace_f32() {
        let mut a = [-3.0f32, -1.0, 0.0, 1.0, 3.0, -5.0, 7.0];
        abs_inplace_f32(&mut a);
        assert_eq!(a, [3.0, 1.0, 0.0, 1.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_recip_inplace_f32() {
        let mut a = [1.0f32, 2.0, 4.0, 0.5, 10.0];
        recip_inplace_f32(&mut a);
        assert_eq!(a, [1.0, 0.5, 0.25, 2.0, 0.1]);
    }

    /// Distance in representable f32 values; both inputs must have the same sign.
    fn ulps(a: f32, b: f32) -> u32 {
        a.to_bits().abs_diff(b.to_bits())
    }

    /// Distance in representable f32 values, across zero as well.
    fn signed_ulps(a: f32, b: f32) -> u64 {
        let ord = |x: f32| {
            let b = x.to_bits() as i32 as i64;
            if b < 0 { i64::from(i32::MIN) - b } else { b }
        };
        ord(a).abs_diff(ord(b))
    }

    /// Checks `op` against `reference` within `max_ulps`: on every 997th bit
    /// pattern up to [`TRIG_MAX`] of both signs and on `extra`, which all
    /// take the SIMD path, then on vectors that mix small lanes with lanes
    /// past the cutoff, which go through libm whole. Whether `TRIG_MAX`
    /// itself is low enough rests on the exhaustive run noted below.
    fn check_trig(op: impl LaneOp<f32>, reference: fn(f32) -> f32, max_ulps: u64, extra: &[f32]) {
        check_within(op, reference, max_ulps, TRIG_MAX, extra);
        check_values(
            op,
            reference,
            max_ulps,
            &[
                0.5,
                TRIG_MAX,
                TRIG_MAX.next_up(),
                1e6,
                -1e6,
                f32::MAX,
                f32::MIN,
                -0.5,
            ],
        );

        // NaN stays on the SIMD path next to ordinary lanes; inf goes through
        // libm. Both give NaN.
        let out = map_to_vec(&[f32::NAN, 1.0, f32::INFINITY, f32::NEG_INFINITY], op);
        assert!(
            out[0].is_nan() && out[2].is_nan() && out[3].is_nan(),
            "{out:?}"
        );
        let out = map_to_vec(&[f32::NAN, 1.0], op);
        assert!(out[0].is_nan(), "{out:?}");
        assert!(signed_ulps(out[1], reference(1.0)) <= max_ulps);
    }

    // The extra inputs are the worst cases of an exhaustive run over
    // [0, TRIG_MAX] on NEON and SSE4.2.

    #[test]
    fn sin_f32_within_2_ulp_of_libm() {
        let extra = [f32::from_bits(0x4155_bc3b), f32::from_bits(0x4069_85b9)];
        check_trig(Sin, libm::sinf, 2, &extra);
        let zeros = map_to_vec(&[0.0, -0.0], Sin);
        assert_eq!(zeros[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(zeros[1].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn cos_f32_within_2_ulp_of_libm() {
        check_trig(Cos, libm::cosf, 2, &[f32::from_bits(0x4016_cc3a)]);
        assert_eq!(map_to_vec(&[0.0, -0.0], Cos), [1.0, 1.0]);
    }

    #[test]
    fn tan_f32_within_4_ulp_of_libm() {
        check_trig(Tan, libm::tanf, 4, &[f32::from_bits(0x4016_d9ff)]);
        let zeros = map_to_vec(&[0.0, -0.0], Tan);
        assert_eq!(zeros[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(zeros[1].to_bits(), (-0.0f32).to_bits());
    }

    /// Checks `op` against `reference` within `max_ulps` on `input`, with NaN
    /// matching NaN.
    fn check_values(op: impl LaneOp<f32>, reference: fn(f32) -> f32, max_ulps: u64, input: &[f32]) {
        let out = map_to_vec(input, op);
        for (&x, &y) in input.iter().zip(&out) {
            let expected = reference(x);
            assert!(
                signed_ulps(y, expected) <= max_ulps || (y.is_nan() && expected.is_nan()),
                "{x:e} gives {y:e}, libm gives {expected:e}"
            );
        }
    }

    /// [`check_values`] on every 997th bit pattern up to `limit` of both
    /// signs, then on `extra` in a call of its own.
    fn check_within(
        op: impl LaneOp<f32>,
        reference: fn(f32) -> f32,
        max_ulps: u64,
        limit: f32,
        extra: &[f32],
    ) {
        let grid: Vec<f32> = (0..=limit.to_bits())
            .step_by(997)
            .flat_map(|b| [f32::from_bits(b), -f32::from_bits(b)])
            .collect();
        check_values(op, reference, max_ulps, &grid);
        check_values(op, reference, max_ulps, extra);
    }

    // The extra inputs include the worst cases of an exhaustive run on NEON
    // and SSE4.2, over every f32 for atan and over [-1, 1] for asin and acos.

    #[test]
    fn atan_f32_within_3_ulp_of_libm() {
        let extra = [
            f32::from_bits(0x3ed6_81fb),
            f32::from_bits(0x3ed6_a91f),
            // The f32 thresholds and the values just above, on either side of
            // each reduction.
            0.414_213_57,
            0.414_213_57f32.next_up(),
            2.414_213_7,
            2.414_213_7f32.next_up(),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        check_within(Atan, libm::atanf, 3, f32::MAX, &extra);
        let zeros = map_to_vec(&[0.0, -0.0], Atan);
        assert_eq!(zeros[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(zeros[1].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn asin_f32_within_2_ulp_of_libm() {
        let extra = [
            f32::from_bits(0x3f00_0010),
            f32::from_bits(0x3f00_0009),
            0.5,
            1.0,
            -1.0,
            1.000_000_1,
            -2.0,
            f32::INFINITY,
            f32::NAN,
        ];
        check_within(Asin, libm::asinf, 2, 1.0, &extra);
        let zeros = map_to_vec(&[0.0, -0.0], Asin);
        assert_eq!(zeros[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(zeros[1].to_bits(), (-0.0f32).to_bits());
    }

    // libm's `acosf` returns the f32 below pi/2 for tiny inputs rather than
    // the nearest one, so most of the range is 1 ulp from it, and the f32
    // below pi for -1.
    #[test]
    fn acos_f32_within_1_ulp_of_libm() {
        let extra = [
            0.5,
            -0.5,
            1.0,
            -1.0,
            1.000_000_1,
            -2.0,
            f32::INFINITY,
            f32::NAN,
        ];
        check_within(Acos, libm::acosf, 1, 1.0, &extra);
        let ends = map_to_vec(&[1.0, -1.0], Acos);
        assert_eq!(ends[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(ends[1].to_bits(), core::f32::consts::PI.to_bits());
    }

    /// Checks the two-input `op` against `reference` within `max_ulps` on
    /// seeded random pairs, uniform over all bit patterns and over [-10, 10],
    /// then on `extra` pairs in a call of their own. Zeros must match in sign.
    fn check_pairs(
        op: impl LaneOp2<f32>,
        reference: fn(f32, f32) -> f32,
        max_ulps: u64,
        extra: &[(f32, f32)],
    ) {
        let check = |xs: &[f32], ys: &[f32]| {
            let out = zip_map_to_vec(xs, ys, op);
            for ((&x, &y), &o) in xs.iter().zip(ys).zip(&out) {
                let e = reference(x, y);
                let ok = if e.is_nan() {
                    o.is_nan()
                } else if e == 0.0 {
                    o.to_bits() == e.to_bits()
                } else {
                    signed_ulps(o, e) <= max_ulps
                };
                assert!(ok, "({x:e}, {y:e}) gives {o:e}, libm gives {e:e}");
            }
        };
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let bits: Vec<f32> = (0..400_000)
            .map(|_| f32::from_bits(next() as u32))
            .collect();
        let small: Vec<f32> = (0..400_000)
            .map(|_| (next() >> 40) as f32 / (1u64 << 24) as f32 * 20.0 - 10.0)
            .collect();
        check(&bits[..200_000], &bits[200_000..]);
        check(&small[..200_000], &small[200_000..]);
        let (xs, ys): (Vec<f32>, Vec<f32>) = extra.iter().copied().unzip();
        check(&xs, &ys);
    }

    /// Every pair of zeros, ones, infinities, NaN and a subnormal, both signs.
    fn special_pairs() -> Vec<(f32, f32)> {
        let v = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(1),
        ];
        v.iter()
            .flat_map(|&a| v.iter().map(move |&b| (a, b)))
            .collect()
    }

    // The extra pairs include the worst cases of 52M random pairs on NEON and
    // SSE4.2.

    #[test]
    fn atan2_f32_within_3_ulp_of_libm() {
        let mut extra = special_pairs();
        extra.extend([
            (f32::from_bits(0x1b4f_0572), f32::from_bits(0x1bed_c8a0)),
            (f32::from_bits(0xc033_4ba6), f32::from_bits(0x40c6_e730)),
            (f32::from_bits(0xc081_c660), f32::from_bits(0x4110_8d10)),
            (f32::MAX, f32::from_bits(1)),
            (f32::from_bits(1), -f32::MAX),
        ]);
        check_pairs(Atan2, libm::atan2f, 3, &extra);
    }

    #[test]
    fn hypot_f32_within_1_ulp_of_libm() {
        let mut extra = special_pairs();
        extra.extend([
            (f32::from_bits(0x0fb2_6aa6), f32::from_bits(0x13ba_2dc9)),
            (f32::from_bits(0xc0b2_87d6), f32::from_bits(0xc081_19b5)),
            (f32::MAX, f32::MAX),
            (f32::MAX, 1.0),
            (3e38, 2e38),
            (f32::from_bits(3), f32::from_bits(4)),
            (1e-30, 1e-30),
        ]);
        check_pairs(Hypot, libm::hypotf, 1, &extra);
    }

    #[test]
    fn exp_f32_within_2_ulp_of_libm() {
        // Every 1/256 step from deep underflow to overflow, which crosses the
        // subnormal range below -87.3 and the overflow edge at 88.72.
        let input: Vec<f32> = (-105 * 256..=90 * 256)
            .map(|i| i as f32 / 256.0)
            .chain([88.72283, 88.72284, -87.33655, -103.27893])
            .collect();
        let mut out = input.clone();
        map_inplace(&mut out, Exp);
        for (&x, &y) in input.iter().zip(&out) {
            let expected = libm::expf(x);
            assert!(
                ulps(y, expected) <= 2,
                "exp({x}) = {y:e}, libm gives {expected:e}"
            );
        }
    }

    #[test]
    fn exp_f32_special_values() {
        let mut a = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            200.0,
            -200.0,
        ];
        map_inplace(&mut a, Exp);
        assert!(a[0].is_nan());
        assert_eq!(&a[1..], &[f32::INFINITY, 0.0, 1.0, 1.0, f32::INFINITY, 0.0]);
    }

    #[test]
    fn sigmoid_f32_matches_stable_scalar() {
        let input: Vec<f32> = (-120 * 16..=120 * 16).map(|i| i as f32 / 16.0).collect();
        let out = map_to_vec(&input, Sigmoid);
        for (&x, &y) in input.iter().zip(&out) {
            let expected = if x >= 0.0 {
                1.0 / (1.0 + libm::expf(-x))
            } else {
                let e = libm::expf(x);
                e / (1.0 + e)
            };
            assert!(
                ulps(y, expected) <= 3,
                "sigmoid({x}) = {y:e}, expected {expected:e}"
            );
        }

        let mut special = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0];
        map_inplace(&mut special, Sigmoid);
        assert!(special[0].is_nan());
        assert_eq!(&special[1..], &[1.0, 0.0, 0.5]);
    }

    #[test]
    fn ln_f32_matches_libm_bitwise() {
        // Every 997th bit pattern from +0 to f32::MAX, subnormals and the
        // [sqrt(2)/2, sqrt(2)) split included, then the special values.
        let input: Vec<f32> = (0..0x7f80_0000u32)
            .step_by(997)
            .map(f32::from_bits)
            .chain([
                f32::from_bits(1),
                f32::MIN_POSITIVE,
                core::f32::consts::FRAC_1_SQRT_2,
                1.0,
                f32::MAX,
            ])
            .collect();
        let out = map_to_vec(&input, Ln);
        for (&x, &y) in input.iter().zip(&out) {
            assert_eq!(y.to_bits(), libm::logf(x).to_bits(), "ln({x:e})");
        }

        let mut special = [f32::NAN, -1.0, f32::NEG_INFINITY, 0.0, -0.0, f32::INFINITY];
        map_inplace(&mut special, Ln);
        assert!(special[..3].iter().all(|y| y.is_nan()), "{special:?}");
        assert_eq!(
            &special[3..],
            &[f32::NEG_INFINITY, f32::NEG_INFINITY, f32::INFINITY]
        );
    }

    /// Checks `op` bit for bit against `reference` on every 1999th bit pattern
    /// of both signs, which crosses each range of the libm routine, then on
    /// `extra`.
    fn check_bitwise(op: impl LaneOp<f32>, reference: fn(f32) -> f32, extra: &[f32]) {
        let input: Vec<f32> = (0..=u32::MAX)
            .step_by(1999)
            .map(f32::from_bits)
            .chain(extra.iter().copied())
            .collect();
        let out = map_to_vec(&input, op);
        for (&x, &y) in input.iter().zip(&out) {
            let expected = reference(x);
            assert!(
                y.to_bits() == expected.to_bits() || (y.is_nan() && expected.is_nan()),
                "{x:e} gives {y:e}, libm gives {expected:e}"
            );
        }
    }

    const INF_NAN_ZEROS: [f32; 5] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];

    #[test]
    fn ln_1p_f32_matches_libm_bitwise() {
        // Both sides of the two cutoffs between libm's reductions.
        let extra = [
            -1.0,
            -0.999_999_94,
            f32::from_bits(0xbe95_f619),
            f32::from_bits(0xbe95_f61a),
            f32::from_bits(0x3ed4_13cf),
            f32::from_bits(0x3ed4_13d0),
            5.9e-8,
            -5.9e-8,
            f32::MAX,
        ];
        check_bitwise(Ln1p, libm::log1pf, &[&INF_NAN_ZEROS[..], &extra].concat());
    }

    #[test]
    fn atanh_f32_matches_libm_bitwise() {
        let extra = [1.0, -1.0, 0.5, -0.5, 0.999_999_94, 1.5, 2.3e-10];
        check_bitwise(Atanh, libm::atanhf, &[&INF_NAN_ZEROS[..], &extra].concat());
    }

    #[test]
    fn log_sigmoid_f32_matches_stable_scalar() {
        // Past +-104, where the exponential underflows and the result is -0
        // or x itself.
        let input: Vec<f32> = (-120 * 256..=120 * 256)
            .map(|i| i as f32 / 256.0)
            .chain([f32::INFINITY, f32::NEG_INFINITY])
            .collect();
        let out = map_to_vec(&input, LogSigmoid);
        let grad = map_to_vec(&input, LogSigmoidGrad);
        for ((&x, &y), &dy) in input.iter().zip(&out).zip(&grad) {
            let expected = if x >= 0.0 {
                -libm::log1pf(libm::expf(-x))
            } else {
                x - libm::log1pf(libm::expf(x))
            };
            assert!(
                ulps(y, expected) <= 1,
                "log_sigmoid({x}) = {y:e}, expected {expected:e}"
            );
            let e = libm::expf(-x.abs());
            let s = 1.0 / (1.0 + e);
            let expected = if x >= 0.0 { e * s } else { s };
            assert!(
                ulps(dy, expected) <= 2,
                "log_sigmoid'({x}) = {dy:e}, expected {expected:e}"
            );
        }
        assert!(map_to_vec(&[f32::NAN], LogSigmoid)[0].is_nan());
    }

    /// Every 1/4096 step over [-12, 12], which crosses each branch and reaches
    /// the range where tanh and erf round to +-1, then tiny values.
    fn odd_grid() -> Vec<f32> {
        (-12 * 4096..=12 * 4096)
            .map(|i| i as f32 / 4096.0)
            .chain([1e-30, -1e-30, f32::from_bits(1), -f32::from_bits(1)])
            .collect()
    }

    /// NaN, signed zeros, and inputs that saturate to +-1, including some
    /// past the grid where x * x overflows for a finite x.
    fn check_odd_saturating_specials<O: LaneOp<f32>>(op: O) {
        let mut special = [
            f32::NAN,
            0.0,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1e20,
            f32::MIN,
        ];
        map_inplace(&mut special, op);
        let [nan, rest @ ..] = special;
        assert!(nan.is_nan());
        assert_eq!(
            rest.map(f32::to_bits),
            [0.0, -0.0, 1.0, -1.0, 1.0, -1.0].map(f32::to_bits)
        );
    }

    #[test]
    fn tanh_f32_within_2_ulp_of_libm() {
        let input = odd_grid();
        let out = map_to_vec(&input, Tanh);
        for (&x, &y) in input.iter().zip(&out) {
            let expected = libm::tanhf(x);
            assert!(
                ulps(y, expected) <= 2,
                "tanh({x}) = {y:e}, libm gives {expected:e}"
            );
        }

        check_odd_saturating_specials(Tanh);
    }

    #[test]
    fn erf_f32_within_1_ulp_of_libm() {
        let input = odd_grid();
        let out = map_to_vec(&input, Erf);
        for (&x, &y) in input.iter().zip(&out) {
            let expected = libm::erff(x);
            assert!(
                ulps(y, expected) <= 1,
                "erf({x}) = {y:e}, libm gives {expected:e}"
            );
        }

        check_odd_saturating_specials(Erf);
    }

    // GELU and its derivative cancel where `1 + erf` nears 0 and where the
    // derivative crosses 0 near x = -0.75, so they are checked by absolute
    // error against f64, scaled by |x| for the large-x range.
    #[test]
    fn gelu_f32_matches_f64_reference() {
        let input = odd_grid();
        let gelu = map_to_vec(&input, Gelu);
        let grad = map_to_vec(&input, GeluGrad);
        for ((&x, &y), &dy) in input.iter().zip(&gelu).zip(&grad) {
            let xd = x as f64;
            let cdf = 0.5 * (1.0 + libm::erf(xd * core::f64::consts::FRAC_1_SQRT_2));
            let pdf = (-0.5 * xd * xd).exp() / (2.0 * core::f64::consts::PI).sqrt();
            let tol = 1e-6 * xd.abs().max(1.0);
            assert!((y as f64 - xd * cdf).abs() <= tol, "gelu({x}) = {y:e}");
            assert!(
                (dy as f64 - (cdf + xd * pdf)).abs() <= tol,
                "gelu'({x}) = {dy:e}"
            );
        }

        // NaN, signed zeros, and finite inputs where x * x overflows, so the
        // density's exp sees -inf and must give 0 rather than NaN.
        let special = [f32::NAN, 0.0, -0.0, 1e20, -1e20];
        let [nan, rest @ ..]: [f32; 5] = map_to_vec(&special, Gelu).try_into().unwrap();
        assert!(nan.is_nan());
        assert_eq!(
            rest.map(f32::to_bits),
            [0.0, -0.0, 1e20, -0.0].map(f32::to_bits)
        );
        let [nan, rest @ ..]: [f32; 5] = map_to_vec(&special, GeluGrad).try_into().unwrap();
        assert!(nan.is_nan());
        assert_eq!(
            rest.map(f32::to_bits),
            [0.5, 0.5, 1.0, 0.0].map(f32::to_bits)
        );
    }

    // Ties, signed zeros, values past the last fractional bit, and NaN/inf,
    // repeated to 69 = 4 * 16 + 5 elements so the unrolled loop runs and a tail
    // is left on every lane width up to 16.
    const ROUND_CASES: [f64; 18] = [
        0.5,
        1.5,
        2.5,
        -0.5,
        -1.5,
        -2.5,
        -0.3,
        0.7,
        -0.0,
        0.0,
        8_388_609.0,
        // Ties just below 2^23 and 2^52, the last values with a half bit in
        // f32 and f64.
        8_388_607.5,
        4_503_599_627_370_495.5,
        4_503_599_627_370_497.0,
        -1.0e300,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];

    fn check_round<O: LaneOp<f32> + LaneOp<f64>>(op: O, scalar_f64: fn(f64) -> f64) {
        let f64_in: Vec<f64> = ROUND_CASES.iter().copied().cycle().take(69).collect();
        let f32_in: Vec<f32> = f64_in.iter().map(|&x| x as f32).collect();

        let mut out = f64_in.clone();
        map_inplace(&mut out, op);
        for (&x, &y) in f64_in.iter().zip(&out) {
            assert_eq!(y.to_bits(), scalar_f64(x).to_bits(), "f64 {x}");
        }

        let out = map_to_vec(&f32_in, op);
        for (&x, &y) in f32_in.iter().zip(&out) {
            let expected = scalar_f64(x as f64) as f32;
            assert_eq!(y.to_bits(), expected.to_bits(), "f32 {x}");
        }
    }

    /// Above `PARALLEL_THRESHOLD`, `map_to_vec` writes rayon chunks into
    /// uninitialized memory; every element must still be written once.
    #[cfg(feature = "rayon")]
    #[test]
    fn map_to_vec_parallel_writes_every_element() {
        let input: Vec<f32> = (0..PARALLEL_THRESHOLD + 3)
            .map(|i| i as f32 * 0.25 - 1000.5)
            .collect();
        let out = map_to_vec(&input, Floor);
        assert_eq!(out.len(), input.len());
        for (i, (&x, &y)) in input.iter().zip(&out).enumerate() {
            assert_eq!(y.to_bits(), x.floor().to_bits(), "index {i}");
        }
    }

    /// Above `PARALLEL_THRESHOLD` both zip maps split into rayon chunks, which
    /// must pair each `a` chunk with its own `b` chunk. Subtraction shows a
    /// swap or a misaligned pair.
    #[cfg(feature = "rayon")]
    #[test]
    fn zip_maps_parallel_pair_every_element() {
        #[derive(Clone, Copy)]
        struct Sub;
        impl LaneOp2<f32> for Sub {
            fn apply<S: Simd>(self, a: Vector<S, f32>, b: Vector<S, f32>) -> Vector<S, f32> {
                a - b
            }
        }

        let n = PARALLEL_THRESHOLD + 3;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let out = zip_map_to_vec(&a, &b, Sub);
        let mut inplace = a.clone();
        zip_map_inplace(&mut inplace, &b, Sub);
        for i in 0..n {
            let expected = a[i] - b[i];
            assert_eq!(out[i], expected, "index {i}");
            assert_eq!(inplace[i], expected, "index {i}");
        }
    }

    #[test]
    fn rounding_matches_scalar_bitwise() {
        check_round(Floor, f64::floor);
        check_round(Ceil, f64::ceil);
        check_round(Trunc, f64::trunc);
        check_round(RoundTiesEven, f64::round_ties_even);
    }

    // ================================================================
    // mask_where / mask_fill tests
    // ================================================================

    #[test]
    fn test_mask_where_f32_basic() {
        let tensor = [1.0f32, 2.0, 3.0, 4.0];
        let mask = [1u8, 0, 1, 0];
        let value = [10.0f32, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 4];
        mask_where_f32(&tensor, &mask, &value, &mut out);
        assert_eq!(out, [10.0, 2.0, 30.0, 4.0]);
    }

    #[test]
    fn test_mask_where_f32_all_true() {
        let tensor = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mask = [1u8; 9];
        let value = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let mut out = [0.0f32; 9];
        mask_where_f32(&tensor, &mask, &value, &mut out);
        assert_eq!(out, value);
    }

    #[test]
    fn test_mask_where_f32_all_false() {
        let tensor = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mask = [0u8; 9];
        let value = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let mut out = [0.0f32; 9];
        mask_where_f32(&tensor, &mask, &value, &mut out);
        assert_eq!(out, tensor);
    }

    #[test]
    fn test_mask_where_f64_basic() {
        let tensor = [1.0f64, 2.0, 3.0];
        let mask = [0u8, 1, 0];
        let value = [10.0f64, 20.0, 30.0];
        let mut out = [0.0f64; 3];
        mask_where_f64(&tensor, &mask, &value, &mut out);
        assert_eq!(out, [1.0, 20.0, 3.0]);
    }

    #[test]
    fn test_mask_where_i64_basic() {
        let tensor = [10i64, 20, 30, 40, 50];
        let mask = [1u8, 0, 1, 0, 1];
        let value = [-1i64, -2, -3, -4, -5];
        let mut out = [0i64; 5];
        mask_where_i64(&tensor, &mask, &value, &mut out);
        assert_eq!(out, [-1, 20, -3, 40, -5]);
    }

    #[test]
    fn test_mask_where_u8_basic() {
        let tensor = [0u8, 1, 0, 1];
        let mask = [1u8, 1, 0, 0];
        let value = [1u8, 0, 1, 0];
        let mut out = [0u8; 4];
        mask_where_u8(&tensor, &mask, &value, &mut out);
        assert_eq!(out, [1, 0, 0, 1]);
    }

    #[test]
    fn test_mask_fill_f32_basic() {
        let tensor = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mask = [1u8, 0, 1, 0, 1, 0, 1];
        let mut out = [0.0f32; 7];
        mask_fill_f32(&tensor, &mask, -1.0, &mut out);
        assert_eq!(out, [-1.0, 2.0, -1.0, 4.0, -1.0, 6.0, -1.0]);
    }

    #[test]
    fn test_mask_fill_f64_basic() {
        let tensor = [1.0f64, 2.0, 3.0];
        let mask = [0u8, 1, 0];
        let mut out = [0.0f64; 3];
        mask_fill_f64(&tensor, &mask, 99.0, &mut out);
        assert_eq!(out, [1.0, 99.0, 3.0]);
    }

    #[test]
    fn test_mask_fill_i64_basic() {
        let tensor = [10i64, 20, 30, 40];
        let mask = [1u8, 0, 0, 1];
        let mut out = [0i64; 4];
        mask_fill_i64(&tensor, &mask, -1, &mut out);
        assert_eq!(out, [-1, 20, 30, -1]);
    }

    #[test]
    fn test_mask_fill_u8_basic() {
        let tensor = [0u8, 1, 0, 1, 0];
        let mask = [1u8, 1, 0, 0, 1];
        let mut out = [0u8; 5];
        mask_fill_u8(&tensor, &mask, 1, &mut out);
        assert_eq!(out, [1, 1, 0, 1, 1]);
    }

    #[test]
    fn test_mask_where_f32_nan() {
        let tensor = [f32::NAN, 2.0, 3.0, f32::NAN];
        let mask = [1u8, 0, 1, 0];
        let value = [10.0f32, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 4];
        mask_where_f32(&tensor, &mask, &value, &mut out);
        // mask=1 picks value, mask=0 picks tensor (including NaN)
        assert_eq!(out[0], 10.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 30.0);
        assert!(out[3].is_nan());
    }

    // Lane-boundary tests: sizes that exercise SIMD + scalar tail on all ISAs.
    // 17 elements for f32 = 4 NEON iters + 1 tail, or 2 AVX2 iters + 1 tail.

    #[test]
    fn test_mask_where_f32_lane_boundary() {
        let n = 17;
        let tensor: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let value: Vec<f32> = (0..n).map(|i| (i as f32) * 10.0).collect();
        let mask: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let mut out = vec![0.0f32; n];
        mask_where_f32(&tensor, &mask, &value, &mut out);
        for i in 0..n {
            let expected = if i % 2 != 0 { value[i] } else { tensor[i] };
            assert_eq!(out[i], expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_mask_fill_f32_lane_boundary() {
        let n = 17;
        let tensor: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mask: Vec<u8> = (0..n).map(|i| (i % 3 == 0) as u8).collect();
        let mut out = vec![0.0f32; n];
        mask_fill_f32(&tensor, &mask, -1.0, &mut out);
        for i in 0..n {
            let expected = if i % 3 == 0 { -1.0 } else { tensor[i] };
            assert_eq!(out[i], expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_mask_where_u8_lane_boundary() {
        // 33 elements for u8: exercises 2 NEON iters + 1 tail, or 1 AVX2 iter + 1 tail
        let n = 33;
        let tensor: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let value: Vec<u8> = (0..n).map(|i| ((i + 1) % 2) as u8).collect();
        let mask: Vec<u8> = (0..n).map(|i| (i % 3 == 0) as u8).collect();
        let mut out = vec![0u8; n];
        mask_where_u8(&tensor, &mask, &value, &mut out);
        for i in 0..n {
            let expected = if i % 3 == 0 { value[i] } else { tensor[i] };
            assert_eq!(out[i], expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_mask_where_f64_lane_boundary() {
        // 9 elements for f64: 4 NEON iters + 1 tail (2 lanes), or 2 AVX2 iters + 1 tail (4 lanes)
        let n = 9;
        let tensor: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let value: Vec<f64> = (0..n).map(|i| -(i as f64)).collect();
        let mask: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let mut out = vec![0.0f64; n];
        mask_where_f64(&tensor, &mask, &value, &mut out);
        for i in 0..n {
            let expected = if i % 2 != 0 { value[i] } else { tensor[i] };
            assert_eq!(out[i], expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_mask_where_empty() {
        let mut out = vec![0.0f32; 0];
        mask_where_f32(&[], &[], &[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_mask_fill_empty() {
        let mut out = vec![0.0f32; 0];
        mask_fill_f32(&[], &[], 1.0, &mut out);
        assert!(out.is_empty());
    }

    // bool_not_u8 writes into a `*mut bool` via macerator's store_as_bool.
    // Rust's bool is only valid as 0x00 or 0x01 (any other byte is UB when
    // read back as bool). A previous audit flagged that SIMD mask stores
    // might emit 0xFF. Verify every output byte is normalized to 0/1.
    #[test]
    fn bool_not_u8_output_is_normalized_0_or_1() {
        // Spans SIMD body + scalar tail on any realistic lane width:
        //   17 elements -> NEON 16-byte SIMD + 1 tail; AVX2 32 spills to tail.
        //   127 elements -> SIMD body + 15 tail for 16-byte lanes.
        for &len in &[1usize, 8, 15, 16, 17, 31, 32, 63, 127, 256] {
            let a: Vec<u8> = (0..len).map(|i| (i % 2) as u8).collect();
            let mut out = vec![0xAAu8; len];
            super::bool_not_u8(&a, &mut out);
            for (i, &b) in out.iter().enumerate() {
                assert!(
                    b == 0 || b == 1,
                    "len={}: out[{}] = 0x{:02x}, expected 0x00 or 0x01",
                    len,
                    i,
                    b
                );
                let expected = if a[i] == 0 { 1 } else { 0 };
                assert_eq!(
                    b, expected,
                    "len={}: out[{}] = {}, expected {}",
                    len, i, b, expected
                );
            }
        }
    }

    #[test]
    fn bool_not_inplace_u8_output_is_normalized_0_or_1() {
        for &len in &[1usize, 8, 15, 16, 17, 31, 32, 63, 127, 256] {
            let mut a: Vec<u8> = (0..len).map(|i| (i % 2) as u8).collect();
            let original = a.clone();
            super::bool_not_inplace_u8(&mut a);
            for (i, &b) in a.iter().enumerate() {
                assert!(
                    b == 0 || b == 1,
                    "len={}: a[{}] = 0x{:02x}, expected 0x00 or 0x01",
                    len,
                    i,
                    b
                );
                let expected = if original[i] == 0 { 1 } else { 0 };
                assert_eq!(
                    b, expected,
                    "len={}: a[{}] = {}, expected {}",
                    len, i, b, expected
                );
            }
        }
    }

    // Edge cases: empty input, homogeneous all-zero, homogeneous all-one.
    // Homogeneous inputs exercise the SIMD mask-to-byte conversion for
    // all-true and all-false cases, which alternating inputs do not.
    #[test]
    fn bool_not_u8_edge_cases() {
        // Empty input.
        let mut out: Vec<u8> = Vec::new();
        super::bool_not_u8(&[], &mut out);
        assert!(out.is_empty());

        // All zeros -> all ones.
        let a = alloc::vec![0u8; 32];
        let mut out = alloc::vec![0xAAu8; 32];
        super::bool_not_u8(&a, &mut out);
        assert!(out.iter().all(|&b| b == 1));

        // All ones -> all zeros.
        let a = alloc::vec![1u8; 32];
        let mut out = alloc::vec![0xAAu8; 32];
        super::bool_not_u8(&a, &mut out);
        assert!(out.iter().all(|&b| b == 0));
    }
}
