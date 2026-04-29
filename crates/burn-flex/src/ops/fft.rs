//! Real FFT (rfft) via Cooley-Tukey with aggressive optimization.
//!
//! Key optimizations:
//! - Real FFT via complex packing: pack N real values as N/2 complex,
//!   do a half-size complex FFT, then unpack using Hermitian symmetry (~2x)
//! - Compile-time twiddle tables via const fn Taylor-series sin/cos
//! - Unrolled small complex FFT kernels for N=2, 4, 8
//! - Mixed radix-4/radix-2 butterfly stages (halves passes over data)
//! - SIMD-vectorized butterflies via macerator
//! - Rayon parallelism across independent fibers

use alloc::vec;
use alloc::vec::Vec;
use burn_std::{Bytes, Shape};

use crate::layout::{contiguous_strides_usize, slice_base_offset};
use crate::{FlexTensor, Layout};

// ============================================================================
// Const-evaluable sin/cos via Taylor series (13 terms, ~13 digit accuracy)
// ============================================================================

const PI: f64 = core::f64::consts::PI;

const fn const_sin(x: f64) -> f64 {
    let mut x = x;
    x = x - ((x / (2.0 * PI)) as i64 as f64) * 2.0 * PI;
    if x > PI {
        x -= 2.0 * PI;
    } else if x < -PI {
        x += 2.0 * PI;
    }
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    let mut i = 1u32;
    while i <= 12 {
        term *= -x2 / ((2 * i) as f64 * (2 * i + 1) as f64);
        sum += term;
        i += 1;
    }
    sum
}

const fn const_cos(x: f64) -> f64 {
    const_sin(x + PI / 2.0)
}

// ============================================================================
// Compile-time twiddle table
// ============================================================================

struct TwiddleTable<const M: usize> {
    re: [f32; M],
    im: [f32; M],
    offsets: [usize; 18],
    num_stages: usize,
}

const fn make_twiddle_table<const N: usize, const M: usize>() -> TwiddleTable<M> {
    let mut re = [0.0f32; M];
    let mut im = [0.0f32; M];
    let mut offsets = [0usize; 18];
    let num_stages = N.trailing_zeros() as usize;
    let mut pos = 0usize;
    let mut len = 2usize;
    let mut stage = 0usize;
    while stage < num_stages {
        offsets[stage] = pos;
        let half = len / 2;
        let angle_step = -2.0 * PI / len as f64;
        let mut k = 0usize;
        while k < half {
            let angle = angle_step * k as f64;
            re[pos] = const_cos(angle) as f32;
            im[pos] = const_sin(angle) as f32;
            pos += 1;
            k += 1;
        }
        len <<= 1;
        stage += 1;
    }
    offsets[num_stages] = pos;
    TwiddleTable {
        re,
        im,
        offsets,
        num_stages,
    }
}

macro_rules! def_twiddle {
    ($name:ident, $n:expr) => {
        static $name: TwiddleTable<{ $n - 1 }> = make_twiddle_table::<$n, { $n - 1 }>();
    };
}

def_twiddle!(TW_2, 2);
def_twiddle!(TW_4, 4);
def_twiddle!(TW_8, 8);
def_twiddle!(TW_16, 16);
def_twiddle!(TW_32, 32);
def_twiddle!(TW_64, 64);
def_twiddle!(TW_128, 128);
def_twiddle!(TW_256, 256);
def_twiddle!(TW_512, 512);
def_twiddle!(TW_1024, 1024);
def_twiddle!(TW_2048, 2048);
def_twiddle!(TW_4096, 4096);
def_twiddle!(TW_8192, 8192);
def_twiddle!(TW_16384, 16384);
def_twiddle!(TW_32768, 32768);
def_twiddle!(TW_65536, 65536);

enum TwiddleRef {
    Static {
        re: &'static [f32],
        im: &'static [f32],
        offsets: &'static [usize],
    },
    Owned {
        re: Vec<f32>,
        im: Vec<f32>,
        offsets: Vec<usize>,
    },
}

impl TwiddleRef {
    fn re(&self) -> &[f32] {
        match self {
            Self::Static { re, .. } => re,
            Self::Owned { re, .. } => re,
        }
    }
    fn im(&self) -> &[f32] {
        match self {
            Self::Static { im, .. } => im,
            Self::Owned { im, .. } => im,
        }
    }
    fn offsets(&self) -> &[usize] {
        match self {
            Self::Static { offsets, .. } => offsets,
            Self::Owned { offsets, .. } => offsets,
        }
    }
}

fn get_twiddles(n: usize) -> TwiddleRef {
    macro_rules! match_static {
        ($($size:expr => $table:ident),+ $(,)?) => {
            match n {
                0 | 1 => TwiddleRef::Static { re: &[], im: &[], offsets: &[0] },
                $($size => TwiddleRef::Static {
                    re: &$table.re, im: &$table.im,
                    offsets: &$table.offsets[..$table.num_stages + 1],
                },)+
                _ => {
                    let (re, im, offsets) = precompute_twiddles_runtime(n);
                    TwiddleRef::Owned { re, im, offsets }
                }
            }
        };
    }
    match_static!(
        2 => TW_2, 4 => TW_4, 8 => TW_8, 16 => TW_16,
        32 => TW_32, 64 => TW_64, 128 => TW_128, 256 => TW_256,
        512 => TW_512, 1024 => TW_1024, 2048 => TW_2048, 4096 => TW_4096,
        8192 => TW_8192, 16384 => TW_16384, 32768 => TW_32768, 65536 => TW_65536,
    )
}

fn precompute_twiddles_runtime(n: usize) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let num_stages = n.trailing_zeros() as usize;
    let total = n - 1;
    let mut re = Vec::with_capacity(total);
    let mut im = Vec::with_capacity(total);
    let mut offsets = Vec::with_capacity(num_stages + 1);
    let mut len = 2;
    for _ in 0..num_stages {
        offsets.push(re.len());
        let half = len / 2;
        let angle_step = -2.0 * core::f64::consts::PI / len as f64;
        for k in 0..half {
            let angle = angle_step * k as f64;
            re.push(const_cos(angle) as f32);
            im.push(const_sin(angle) as f32);
        }
        len <<= 1;
    }
    offsets.push(re.len());
    (re, im, offsets)
}

// ============================================================================
// Bit-reversal permutation
// ============================================================================

#[inline]
fn bit_reverse_permute(re: &mut [f32], im: &mut [f32], n: usize) {
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
}

// ============================================================================
// Unrolled small complex FFT kernels
// ============================================================================

/// Complex FFT of size 2: single butterfly, no twiddles.
#[inline(always)]
fn complex_fft_2(re: &mut [f32], im: &mut [f32]) {
    let (r0, r1) = (re[0], re[1]);
    let (i0, i1) = (im[0], im[1]);
    re[0] = r0 + r1;
    re[1] = r0 - r1;
    im[0] = i0 + i1;
    im[1] = i0 - i1;
}

/// Complex FFT of size 4: 2 stages, fully unrolled.
/// Twiddle for stage 1, k=1 is W_4^1 = -i.
#[inline(always)]
fn complex_fft_4(re: &mut [f32], im: &mut [f32]) {
    // Bit-reversal: swap indices 1 and 2
    re.swap(1, 2);
    im.swap(1, 2);

    // Stage 0: two size-2 butterflies
    let (r0, r1) = (re[0] + re[1], re[0] - re[1]);
    let (i0, i1) = (im[0] + im[1], im[0] - im[1]);
    let (r2, r3) = (re[2] + re[3], re[2] - re[3]);
    let (i2, i3) = (im[2] + im[3], im[2] - im[3]);

    // Stage 1: size-4 butterfly
    // k=0: W=1, butterfly (0,2)
    re[0] = r0 + r2;
    im[0] = i0 + i2;
    re[2] = r0 - r2;
    im[2] = i0 - i2;
    // k=1: W=-i, butterfly (1,3): -i*(r3+i*i3) = (i3, -r3)
    re[1] = r1 + i3;
    im[1] = i1 - r3;
    re[3] = r1 - i3;
    im[3] = i1 + r3;
}

/// Complex FFT of size 8: 3 stages, fully unrolled.
#[inline(always)]
fn complex_fft_8(re: &mut [f32], im: &mut [f32]) {
    // Bit-reversal for n=8: [0,4,2,6,1,5,3,7]
    re.swap(1, 4);
    im.swap(1, 4);
    re.swap(3, 6);
    im.swap(3, 6);

    // Stage 0: four size-2 butterflies
    macro_rules! butterfly2 {
        ($a:expr, $b:expr) => {
            let (ra, rb) = (re[$a] + re[$b], re[$a] - re[$b]);
            let (ia, ib) = (im[$a] + im[$b], im[$a] - im[$b]);
            re[$a] = ra;
            re[$b] = rb;
            im[$a] = ia;
            im[$b] = ib;
        };
    }
    butterfly2!(0, 1);
    butterfly2!(2, 3);
    butterfly2!(4, 5);
    butterfly2!(6, 7);

    // Stage 1: two size-4 butterflies
    // Group [0,1,2,3]: k=0 W=1, k=1 W=-i
    {
        let (r0, r2) = (re[0] + re[2], re[0] - re[2]);
        let (i0, i2) = (im[0] + im[2], im[0] - im[2]);
        re[0] = r0;
        im[0] = i0;
        re[2] = r2;
        im[2] = i2;
        // k=1: W=-i → (im[3], -re[3])
        let (t_re, t_im) = (im[3], -re[3]);
        let (r1a, r1b) = (re[1] + t_re, re[1] - t_re);
        let (i1a, i1b) = (im[1] + t_im, im[1] - t_im);
        re[1] = r1a;
        re[3] = r1b;
        im[1] = i1a;
        im[3] = i1b;
    }
    // Group [4,5,6,7]: same pattern
    {
        let (r4, r6) = (re[4] + re[6], re[4] - re[6]);
        let (i4, i6) = (im[4] + im[6], im[4] - im[6]);
        re[4] = r4;
        im[4] = i4;
        re[6] = r6;
        im[6] = i6;
        let (t_re, t_im) = (im[7], -re[7]);
        let (r5a, r5b) = (re[5] + t_re, re[5] - t_re);
        let (i5a, i5b) = (im[5] + t_im, im[5] - t_im);
        re[5] = r5a;
        re[7] = r5b;
        im[5] = i5a;
        im[7] = i5b;
    }

    // Stage 2: one size-8 butterfly
    // k=0: W=1
    {
        let (a, b) = (re[0] + re[4], re[0] - re[4]);
        let (c, d) = (im[0] + im[4], im[0] - im[4]);
        re[0] = a;
        re[4] = b;
        im[0] = c;
        im[4] = d;
    }
    // k=1: W_8^1 = (sqrt2/2, -sqrt2/2)
    {
        const W: f32 = core::f32::consts::FRAC_1_SQRT_2; // 0.7071...
        let t_re = W * re[5] - (-W) * im[5]; // W*re + W*im
        let t_im = W * im[5] + (-W) * re[5]; // W*im - W*re
        re[5] = re[1] - t_re;
        im[5] = im[1] - t_im;
        re[1] += t_re;
        im[1] += t_im;
    }
    // k=2: W_8^2 = -i
    {
        let (t_re, t_im) = (im[6], -re[6]);
        re[6] = re[2] - t_re;
        im[6] = im[2] - t_im;
        re[2] += t_re;
        im[2] += t_im;
    }
    // k=3: W_8^3 = (-sqrt2/2, -sqrt2/2)
    {
        const W: f32 = core::f32::consts::FRAC_1_SQRT_2;
        let t_re = -W * re[7] - (-W) * im[7]; // -W*re + W*im
        let t_im = -W * im[7] + (-W) * re[7]; // -W*im - W*re
        re[7] = re[3] - t_re;
        im[7] = im[3] - t_im;
        re[3] += t_re;
        im[3] += t_im;
    }
}

// ============================================================================
// General complex FFT: mixed radix-4/radix-2 with SIMD
// ============================================================================

/// Complex FFT of size n (power of 2) using precomputed twiddles.
#[inline]
fn complex_fft(re: &mut [f32], im: &mut [f32], n: usize, tw: &TwiddleRef) {
    match n {
        0 | 1 => return,
        2 => {
            complex_fft_2(re, im);
            return;
        }
        4 => {
            complex_fft_4(re, im);
            return;
        }
        8 => {
            complex_fft_8(re, im);
            return;
        }
        _ => {}
    }

    bit_reverse_permute(re, im, n);

    let tw_re = tw.re();
    let tw_im = tw.im();
    let offsets = tw.offsets();
    let num_stages = offsets.len() - 1;

    // For odd number of stages, do one radix-2 pass first so the
    // remaining stages can be processed in radix-4 pairs.
    // Stage 0 twiddle is always W_2^0 = 1, so just add/sub.
    let start_stage = if num_stages % 2 == 1 {
        let mut start = 0;
        while start < n {
            let (a, b) = (re[start] + re[start + 1], re[start] - re[start + 1]);
            let (c, d) = (im[start] + im[start + 1], im[start] - im[start + 1]);
            re[start] = a;
            re[start + 1] = b;
            im[start] = c;
            im[start + 1] = d;
            start += 2;
        }
        1
    } else {
        0
    };

    #[cfg(feature = "simd")]
    {
        simd_fft::radix4_simd(re, im, n, tw_re, tw_im, offsets, start_stage, num_stages);
    }
    #[cfg(not(feature = "simd"))]
    {
        radix4_scalar(re, im, n, tw_re, tw_im, offsets, start_stage, num_stages);
    }
}

/// Correct DIT radix-4: fuse two radix-2 stages into one pass.
///
/// For each pair of stages (s, s+1) with quarter = 2^s:
/// 1. Apply W_{2q}^k to x[p1] and x[p3] (inner stage twiddle, same for both)
/// 2. Inner butterflies: a=p0+tw1, b=p0-tw1, c=p2+tw3, d=p2-tw3
/// 3. Apply W_{4q}^k to c and d; d also gets -i rotation
/// 4. Outer butterflies: p0=a+tc, p2=a-tc, p1=b+(-i*td), p3=b-(-i*td)
#[cfg(not(feature = "simd"))]
#[allow(clippy::too_many_arguments)]
fn radix4_scalar(
    re: &mut [f32],
    im: &mut [f32],
    n: usize,
    tw_re: &[f32],
    tw_im: &[f32],
    offsets: &[usize],
    start_stage: usize,
    num_stages: usize,
) {
    let mut stage = start_stage;
    while stage + 1 < num_stages {
        let quarter = 1 << stage;
        let group_size = quarter << 2;
        let tw_off_inner = offsets[stage]; // W_{2q}^k
        let tw_off_outer = offsets[stage + 1]; // W_{4q}^k

        let mut group_start = 0;
        while group_start < n {
            for k in 0..quarter {
                let p0 = group_start + k;
                let p1 = p0 + quarter;
                let p2 = p1 + quarter;
                let p3 = p2 + quarter;

                // Inner twiddle: W_{2q}^k applied to p1 and p3
                let wi_re = tw_re[tw_off_inner + k];
                let wi_im = tw_im[tw_off_inner + k];
                let tw1_re = wi_re * re[p1] - wi_im * im[p1];
                let tw1_im = wi_re * im[p1] + wi_im * re[p1];
                let tw3_re = wi_re * re[p3] - wi_im * im[p3];
                let tw3_im = wi_re * im[p3] + wi_im * re[p3];

                // Inner butterflies
                let a_re = re[p0] + tw1_re;
                let a_im = im[p0] + tw1_im;
                let b_re = re[p0] - tw1_re;
                let b_im = im[p0] - tw1_im;
                let c_re = re[p2] + tw3_re;
                let c_im = im[p2] + tw3_im;
                let d_re = re[p2] - tw3_re;
                let d_im = im[p2] - tw3_im;

                // Outer twiddle: W_{4q}^k applied to c and d
                let wo_re = tw_re[tw_off_outer + k];
                let wo_im = tw_im[tw_off_outer + k];
                let tc_re = wo_re * c_re - wo_im * c_im;
                let tc_im = wo_re * c_im + wo_im * c_re;
                let td_re = wo_re * d_re - wo_im * d_im;
                let td_im = wo_re * d_im + wo_im * d_re;

                // Outer butterflies (-i*(td_re+i*td_im) = (td_im, -td_re))
                re[p0] = a_re + tc_re;
                im[p0] = a_im + tc_im;
                re[p2] = a_re - tc_re;
                im[p2] = a_im - tc_im;
                re[p1] = b_re + td_im;
                im[p1] = b_im - td_re;
                re[p3] = b_re - td_im;
                im[p3] = b_im + td_re;
            }
            group_start += group_size;
        }
        stage += 2;
    }
}

#[cfg(feature = "simd")]
mod simd_fft {
    use macerator::{Simd, vload_unaligned, vstore_unaligned};

    /// Scalar radix-4 butterfly for the SIMD tail path.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn scalar_radix4(
        re: &mut [f32],
        im: &mut [f32],
        p0: usize,
        quarter: usize,
        tw_re: &[f32],
        tw_im: &[f32],
        tw_off_inner: usize,
        tw_off_outer: usize,
        k: usize,
    ) {
        let p1 = p0 + quarter;
        let p2 = p1 + quarter;
        let p3 = p2 + quarter;

        let wi_r = tw_re[tw_off_inner + k];
        let wi_i = tw_im[tw_off_inner + k];
        let tw1_re = wi_r * re[p1] - wi_i * im[p1];
        let tw1_im = wi_r * im[p1] + wi_i * re[p1];
        let tw3_re = wi_r * re[p3] - wi_i * im[p3];
        let tw3_im = wi_r * im[p3] + wi_i * re[p3];

        let a_re = re[p0] + tw1_re;
        let a_im = im[p0] + tw1_im;
        let b_re = re[p0] - tw1_re;
        let b_im = im[p0] - tw1_im;
        let c_re = re[p2] + tw3_re;
        let c_im = im[p2] + tw3_im;
        let d_re = re[p2] - tw3_re;
        let d_im = im[p2] - tw3_im;

        let wo_r = tw_re[tw_off_outer + k];
        let wo_i = tw_im[tw_off_outer + k];
        let tc_re = wo_r * c_re - wo_i * c_im;
        let tc_im = wo_r * c_im + wo_i * c_re;
        let td_re = wo_r * d_re - wo_i * d_im;
        let td_im = wo_r * d_im + wo_i * d_re;

        re[p0] = a_re + tc_re;
        im[p0] = a_im + tc_im;
        re[p2] = a_re - tc_re;
        im[p2] = a_im - tc_im;
        re[p1] = b_re + td_im;
        im[p1] = b_im - td_re;
        re[p3] = b_re - td_im;
        im[p3] = b_im + td_re;
    }

    /// SIMD radix-4 butterfly passes (pairs of radix-2 stages).
    #[macerator::with_simd]
    #[allow(clippy::too_many_arguments)]
    pub fn radix4_simd<S: Simd>(
        re: &mut [f32],
        im: &mut [f32],
        n: usize,
        tw_re: &[f32],
        tw_im: &[f32],
        offsets: &[usize],
        start_stage: usize,
        num_stages: usize,
    ) {
        let lanes = S::lanes32();
        let mut stage = start_stage;

        while stage + 1 < num_stages {
            let quarter = 1 << stage;
            let group_size = quarter << 2;
            let tw_off_inner = offsets[stage];
            let tw_off_outer = offsets[stage + 1];

            if quarter >= lanes {
                let mut group_start = 0;
                while group_start < n {
                    let mut k = 0;
                    while k + lanes <= quarter {
                        unsafe {
                            // Inner twiddle: W_{2q}^k
                            let wi_r =
                                vload_unaligned::<S, f32>(tw_re.as_ptr().add(tw_off_inner + k));
                            let wi_i =
                                vload_unaligned::<S, f32>(tw_im.as_ptr().add(tw_off_inner + k));

                            let p0 = group_start + k;
                            let p1 = p0 + quarter;
                            let p2 = p1 + quarter;
                            let p3 = p2 + quarter;

                            let r0 = vload_unaligned::<S, f32>(re.as_ptr().add(p0));
                            let i0 = vload_unaligned::<S, f32>(im.as_ptr().add(p0));
                            let r1 = vload_unaligned::<S, f32>(re.as_ptr().add(p1));
                            let i1 = vload_unaligned::<S, f32>(im.as_ptr().add(p1));
                            let r2 = vload_unaligned::<S, f32>(re.as_ptr().add(p2));
                            let i2 = vload_unaligned::<S, f32>(im.as_ptr().add(p2));
                            let r3 = vload_unaligned::<S, f32>(re.as_ptr().add(p3));
                            let i3 = vload_unaligned::<S, f32>(im.as_ptr().add(p3));

                            // Apply inner twiddle to p1 and p3
                            let tw1_re = wi_r * r1 - wi_i * i1;
                            let tw1_im = wi_r * i1 + wi_i * r1;
                            let tw3_re = wi_r * r3 - wi_i * i3;
                            let tw3_im = wi_r * i3 + wi_i * r3;

                            // Inner butterflies
                            let a_re = r0 + tw1_re;
                            let a_im = i0 + tw1_im;
                            let b_re = r0 - tw1_re;
                            let b_im = i0 - tw1_im;
                            let c_re = r2 + tw3_re;
                            let c_im = i2 + tw3_im;
                            let d_re = r2 - tw3_re;
                            let d_im = i2 - tw3_im;

                            // Outer twiddle: W_{4q}^k
                            let wo_r =
                                vload_unaligned::<S, f32>(tw_re.as_ptr().add(tw_off_outer + k));
                            let wo_i =
                                vload_unaligned::<S, f32>(tw_im.as_ptr().add(tw_off_outer + k));
                            let tc_re = wo_r * c_re - wo_i * c_im;
                            let tc_im = wo_r * c_im + wo_i * c_re;
                            let td_re = wo_r * d_re - wo_i * d_im;
                            let td_im = wo_r * d_im + wo_i * d_re;

                            // Outer butterflies: -i*(td_re+i*td_im) = (td_im, -td_re)
                            vstore_unaligned::<S, f32>(re.as_mut_ptr().add(p0), a_re + tc_re);
                            vstore_unaligned::<S, f32>(im.as_mut_ptr().add(p0), a_im + tc_im);
                            vstore_unaligned::<S, f32>(re.as_mut_ptr().add(p2), a_re - tc_re);
                            vstore_unaligned::<S, f32>(im.as_mut_ptr().add(p2), a_im - tc_im);
                            vstore_unaligned::<S, f32>(re.as_mut_ptr().add(p1), b_re + td_im);
                            vstore_unaligned::<S, f32>(im.as_mut_ptr().add(p1), b_im - td_re);
                            vstore_unaligned::<S, f32>(re.as_mut_ptr().add(p3), b_re - td_im);
                            vstore_unaligned::<S, f32>(im.as_mut_ptr().add(p3), b_im + td_re);
                        }
                        k += lanes;
                    }
                    while k < quarter {
                        scalar_radix4(
                            re,
                            im,
                            group_start + k,
                            quarter,
                            tw_re,
                            tw_im,
                            tw_off_inner,
                            tw_off_outer,
                            k,
                        );
                        k += 1;
                    }
                    group_start += group_size;
                }
            } else {
                let mut group_start = 0;
                while group_start < n {
                    for k in 0..quarter {
                        scalar_radix4(
                            re,
                            im,
                            group_start + k,
                            quarter,
                            tw_re,
                            tw_im,
                            tw_off_inner,
                            tw_off_outer,
                            k,
                        );
                    }
                    group_start += group_size;
                }
            }
            stage += 2;
        }
    }
}

// ============================================================================
// Real FFT unpacking
// ============================================================================

/// Unpack N/2-point complex FFT result into N/2+1 real FFT bins.
///
/// Given Z = FFT(pack(x)), recovers X = FFT(x) using:
///   Xe[k] = (Z[k] + conj(Z[N/2-k])) / 2
///   Xo[k] = -i * (Z[k] - conj(Z[N/2-k])) / 2
///   X[k]  = Xe[k] + W_N^k * Xo[k]
fn unpack_rfft(
    z_re: &[f32],
    z_im: &[f32],
    half: usize,
    unpack_tw_re: &[f32],
    unpack_tw_im: &[f32],
    out_re: &mut [f32],
    out_im: &mut [f32],
) {
    // k=0: X[0] = Z_re[0] + Z_im[0] (real)
    out_re[0] = z_re[0] + z_im[0];
    out_im[0] = 0.0;

    // k=N/2: X[N/2] = Z_re[0] - Z_im[0] (real)
    out_re[half] = z_re[0] - z_im[0];
    out_im[half] = 0.0;

    // k=1..half-1
    for k in 1..half {
        let j = half - k;
        let (zk_re, zk_im) = (z_re[k], z_im[k]);
        let (zj_re, zj_im) = (z_re[j], z_im[j]);

        // Xe = (Z[k] + conj(Z[j])) / 2
        let xe_re = (zk_re + zj_re) * 0.5;
        let xe_im = (zk_im - zj_im) * 0.5;

        // Xo = -i * (Z[k] - conj(Z[j])) / 2
        // diff = Z[k] - conj(Z[j]) = (zk_re - zj_re, zk_im + zj_im)
        // -i * diff = (diff_im, -diff_re)
        let xo_re = (zk_im + zj_im) * 0.5;
        let xo_im = (zj_re - zk_re) * 0.5;

        // X[k] = Xe + W * Xo
        let wr = unpack_tw_re[k];
        let wi = unpack_tw_im[k];

        out_re[k] = xe_re + wr * xo_re - wi * xo_im;
        out_im[k] = xe_im + wr * xo_im + wi * xo_re;
    }
}

// ============================================================================
// Tensor helpers
// ============================================================================

fn make_tensors_typed<E: burn_backend::Element + bytemuck::Pod>(
    re: Vec<E>,
    im: Vec<E>,
    shape: Shape,
) -> (FlexTensor, FlexTensor) {
    let dtype = E::dtype();
    let re_t = FlexTensor::new(
        Bytes::from_elems(re),
        Layout::contiguous(shape.clone()),
        dtype,
    );
    let im_t = FlexTensor::new(Bytes::from_elems(im), Layout::contiguous(shape), dtype);
    (re_t, im_t)
}

// ============================================================================
// Top-level rfft: real FFT via complex packing
// ============================================================================

/// Process a single fiber: pack real signal as complex, FFT, unpack.
#[allow(clippy::too_many_arguments)]
#[inline]
fn rfft_fiber(
    signal: &[f32],
    in_stride: usize,
    n: usize,
    sig_len: usize,
    out_re: &mut [f32],
    out_im: &mut [f32],
    tw_half: &TwiddleRef,
    unpack_tw_re: &[f32],
    unpack_tw_im: &[f32],
    z_re: &mut [f32],
    z_im: &mut [f32],
) {
    let half = n / 2;

    if n == 1 {
        out_re[0] = if sig_len >= 1 { signal[0] } else { 0.0 };
        out_im[0] = 0.0;
        return;
    }

    if sig_len >= n {
        if in_stride == 1 {
            for k in 0..half {
                z_re[k] = signal[2 * k];
                z_im[k] = signal[2 * k + 1];
            }
        } else {
            for k in 0..half {
                z_re[k] = signal[(2 * k) * in_stride];
                z_im[k] = signal[(2 * k + 1) * in_stride];
            }
        }
    } else {
        for k in 0..half {
            let even = 2 * k;
            let odd = 2 * k + 1;
            z_re[k] = if even < sig_len {
                signal[even * in_stride]
            } else {
                0.0
            };
            z_im[k] = if odd < sig_len {
                signal[odd * in_stride]
            } else {
                0.0
            };
        }
    }

    complex_fft(z_re, z_im, half, tw_half);
    unpack_rfft(z_re, z_im, half, unpack_tw_re, unpack_tw_im, out_re, out_im);
}

pub fn rfft_f32(tensor: FlexTensor, dim: usize, n: Option<usize>) -> (FlexTensor, FlexTensor) {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    assert!(
        dim < shape.num_dims(),
        "rfft: dim {dim} out of bounds for {}-D tensor",
        shape.num_dims()
    );

    let requested_n = n.unwrap_or_else(|| {
        let sig_len = shape[dim];
        assert!(
            sig_len > 0 && sig_len.is_power_of_two(),
            "rfft: dimension size must be a power of 2, got {sig_len}"
        );
        sig_len
    });
    let fft_size = requested_n.next_power_of_two();
    let sig_len = shape[dim].min(requested_n);

    let n = fft_size;
    let out_len = n / 2 + 1;

    let mut out_dims: Vec<usize> = shape.as_slice().to_vec();
    out_dims[dim] = out_len;
    let out_shape = Shape::from(out_dims);
    let total_out = out_shape.num_elements();
    let num_fibers = shape.num_elements() / shape[dim];

    let data: &[f32] = tensor.storage();
    let in_strides = contiguous_strides_usize(&shape);
    let out_strides = contiguous_strides_usize(&out_shape);

    // N=1: each element is its own DFT, no twiddles needed
    if n == 1 {
        let mut re_out = vec![0.0f32; total_out];
        let im_out = vec![0.0f32; total_out];
        if sig_len >= 1 {
            for fiber_idx in 0..num_fibers {
                let base = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
                let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);
                re_out[out_base] = data[base];
            }
        }
        return make_tensors_typed(re_out, im_out, out_shape);
    }

    let half = n / 2;
    let tw_half = get_twiddles(half);

    let tw_full = get_twiddles(n);
    let full_offsets = tw_full.offsets();
    let last_stage_off = if full_offsets.len() >= 2 {
        full_offsets[full_offsets.len() - 2]
    } else {
        0
    };
    let unpack_tw_re = &tw_full.re()[last_stage_off..];
    let unpack_tw_im = &tw_full.im()[last_stage_off..];

    let mut re_out = vec![0.0f32; total_out];
    let mut im_out = vec![0.0f32; total_out];

    let in_stride = in_strides[dim];
    let out_stride = out_strides[dim];

    #[cfg(feature = "rayon")]
    if num_fibers >= 4 && n >= 64 {
        use rayon::prelude::*;

        let fiber_results: Vec<(usize, Vec<f32>, Vec<f32>)> = (0..num_fibers)
            .into_par_iter()
            .map(|fiber_idx| {
                let base_offset = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
                let mut z_re = vec![0.0f32; half.max(1)];
                let mut z_im = vec![0.0f32; half.max(1)];
                let mut fiber_re = vec![0.0f32; out_len];
                let mut fiber_im = vec![0.0f32; out_len];

                rfft_fiber(
                    &data[base_offset..],
                    in_stride,
                    n,
                    sig_len,
                    &mut fiber_re,
                    &mut fiber_im,
                    &tw_half,
                    unpack_tw_re,
                    unpack_tw_im,
                    &mut z_re,
                    &mut z_im,
                );
                (fiber_idx, fiber_re, fiber_im)
            })
            .collect();

        for (fiber_idx, fiber_re, fiber_im) in fiber_results {
            let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);
            for k in 0..out_len {
                re_out[out_base + k * out_stride] = fiber_re[k];
                im_out[out_base + k * out_stride] = fiber_im[k];
            }
        }

        let (re, im) = make_tensors_typed(re_out, im_out, out_shape);
        return (re, im);
    }

    let mut z_re_buf = vec![0.0f32; half.max(1)];
    let mut z_im_buf = vec![0.0f32; half.max(1)];
    let mut fiber_re = vec![0.0f32; out_len];
    let mut fiber_im = vec![0.0f32; out_len];

    for fiber_idx in 0..num_fibers {
        let base_offset = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
        let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);

        rfft_fiber(
            &data[base_offset..],
            in_stride,
            n,
            sig_len,
            &mut fiber_re,
            &mut fiber_im,
            &tw_half,
            unpack_tw_re,
            unpack_tw_im,
            &mut z_re_buf,
            &mut z_im_buf,
        );

        for k in 0..out_len {
            re_out[out_base + k * out_stride] = fiber_re[k];
            im_out[out_base + k * out_stride] = fiber_im[k];
        }
    }

    let (re, im) = make_tensors_typed(re_out, im_out, out_shape);
    (re, im)
}

#[allow(clippy::too_many_arguments)]
fn rfft_fiber_f64(
    signal: &[f64],
    in_stride: usize,
    n: usize,
    sig_len: usize,
    half: usize,
    out_re: &mut [f64],
    out_im: &mut [f64],
    tw_re: &[f32],
    tw_im: &[f32],
    tw_offsets: &[usize],
    unpack_re: &[f32],
    unpack_im: &[f32],
    z_re: &mut [f64],
    z_im: &mut [f64],
) {
    if n == 1 {
        out_re[0] = if sig_len >= 1 { signal[0] } else { 0.0 };
        out_im[0] = 0.0;
        return;
    }

    if sig_len >= n {
        for k in 0..half {
            z_re[k] = signal[(2 * k) * in_stride];
            z_im[k] = signal[(2 * k + 1) * in_stride];
        }
    } else {
        for k in 0..half {
            let even = 2 * k;
            let odd = 2 * k + 1;
            z_re[k] = if even < sig_len {
                signal[even * in_stride]
            } else {
                0.0
            };
            z_im[k] = if odd < sig_len {
                signal[odd * in_stride]
            } else {
                0.0
            };
        }
    }

    fft_f64_inplace(z_re, z_im, half, tw_re, tw_im, tw_offsets);

    out_re[0] = z_re[0] + z_im[0];
    out_im[0] = 0.0;
    out_re[half] = z_re[0] - z_im[0];
    out_im[half] = 0.0;

    for k in 1..half {
        let j = half - k;
        let (zk_re, zk_im) = (z_re[k], z_im[k]);
        let (zj_re, zj_im) = (z_re[j], z_im[j]);

        let xe_re = (zk_re + zj_re) * 0.5;
        let xe_im = (zk_im - zj_im) * 0.5;
        let xo_re = (zk_im + zj_im) * 0.5;
        let xo_im = (zj_re - zk_re) * 0.5;

        let wr = unpack_re[k] as f64;
        let wi = unpack_im[k] as f64;

        out_re[k] = xe_re + wr * xo_re - wi * xo_im;
        out_im[k] = xe_im + wr * xo_im + wi * xo_re;
    }
}

pub fn rfft_f64(tensor: FlexTensor, dim: usize, n: Option<usize>) -> (FlexTensor, FlexTensor) {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    assert!(
        dim < shape.num_dims(),
        "rfft: dim {dim} out of bounds for {}-D tensor",
        shape.num_dims()
    );

    let requested_n = n.unwrap_or_else(|| {
        let sig_len = shape[dim];
        assert!(
            sig_len > 0 && sig_len.is_power_of_two(),
            "rfft: dimension size must be a power of 2, got {sig_len}"
        );
        sig_len
    });
    let fft_size = requested_n.next_power_of_two();
    let sig_len = shape[dim].min(requested_n);

    let n = fft_size;
    let out_len = n / 2 + 1;

    let mut out_dims: Vec<usize> = shape.as_slice().to_vec();
    out_dims[dim] = out_len;
    let out_shape = Shape::from(out_dims);
    let total_out = out_shape.num_elements();
    let num_fibers = shape.num_elements() / shape[dim];

    let data: &[f64] = tensor.storage();
    let in_strides = contiguous_strides_usize(&shape);
    let out_strides = contiguous_strides_usize(&out_shape);
    let half = n / 2;

    let tw_half = get_twiddles(half);
    let tw_full = get_twiddles(n);
    let full_offsets = tw_full.offsets();
    let last_stage_off = if full_offsets.len() >= 2 {
        full_offsets[full_offsets.len() - 2]
    } else {
        0
    };
    let unpack_re = &tw_full.re()[last_stage_off..];
    let unpack_im = &tw_full.im()[last_stage_off..];

    let mut re_out = vec![0.0f64; total_out];
    let mut im_out = vec![0.0f64; total_out];
    let in_stride = in_strides[dim];
    let out_stride = out_strides[dim];

    let tw_half_re = tw_half.re();
    let tw_half_im = tw_half.im();
    let tw_half_offsets = tw_half.offsets();

    #[cfg(feature = "rayon")]
    if num_fibers >= 4 && n >= 64 {
        use rayon::prelude::*;

        let fiber_results: Vec<(usize, Vec<f64>, Vec<f64>)> = (0..num_fibers)
            .into_par_iter()
            .map(|fiber_idx| {
                let base_offset = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
                let mut z_re = vec![0.0f64; half.max(1)];
                let mut z_im = vec![0.0f64; half.max(1)];
                let mut fiber_re = vec![0.0f64; out_len];
                let mut fiber_im = vec![0.0f64; out_len];

                rfft_fiber_f64(
                    &data[base_offset..],
                    in_stride,
                    n,
                    sig_len,
                    half,
                    &mut fiber_re,
                    &mut fiber_im,
                    tw_half_re,
                    tw_half_im,
                    tw_half_offsets,
                    unpack_re,
                    unpack_im,
                    &mut z_re,
                    &mut z_im,
                );
                (fiber_idx, fiber_re, fiber_im)
            })
            .collect();

        for (fiber_idx, fiber_re, fiber_im) in fiber_results {
            let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);
            for k in 0..out_len {
                re_out[out_base + k * out_stride] = fiber_re[k];
                im_out[out_base + k * out_stride] = fiber_im[k];
            }
        }

        let (re, im) = make_tensors_typed(re_out, im_out, out_shape);
        return (re, im);
    }

    let mut z_re = vec![0.0f64; half.max(1)];
    let mut z_im = vec![0.0f64; half.max(1)];
    let mut fiber_re = vec![0.0f64; out_len];
    let mut fiber_im = vec![0.0f64; out_len];

    for fiber_idx in 0..num_fibers {
        let base_offset = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
        let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);

        rfft_fiber_f64(
            &data[base_offset..],
            in_stride,
            n,
            sig_len,
            half,
            &mut fiber_re,
            &mut fiber_im,
            tw_half_re,
            tw_half_im,
            tw_half_offsets,
            unpack_re,
            unpack_im,
            &mut z_re,
            &mut z_im,
        );

        for k in 0..out_len {
            re_out[out_base + k * out_stride] = fiber_re[k];
            im_out[out_base + k * out_stride] = fiber_im[k];
        }
    }

    let (re, im) = make_tensors_typed(re_out, im_out, out_shape);
    (re, im)
}

/// f64 complex FFT using f32 twiddle table (widened in inner loop).
/// Twiddle precision is limited to ~7 digits (f32), so output accuracy
/// is below full f64 precision for large N.
fn fft_f64_inplace(
    re: &mut [f64],
    im: &mut [f64],
    n: usize,
    tw_re: &[f32],
    tw_im: &[f32],
    offsets: &[usize],
) {
    if n <= 1 {
        return;
    }

    // Bit-reversal
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Scalar radix-2 passes
    let num_stages = offsets.len() - 1;
    let mut len = 2;
    for &tw_off in &offsets[..num_stages] {
        let half = len / 2;
        let mut start = 0;
        while start < n {
            for k in 0..half {
                let wr = tw_re[tw_off + k] as f64;
                let wi = tw_im[tw_off + k] as f64;
                let even = start + k;
                let odd = even + half;
                let t_re = wr * re[odd] - wi * im[odd];
                let t_im = wr * im[odd] + wi * re[odd];
                re[odd] = re[even] - t_re;
                im[odd] = im[even] - t_im;
                re[even] += t_re;
                im[even] += t_im;
            }
            start += len;
        }
        len <<= 1;
    }
}

pub fn rfft_f16(tensor: FlexTensor, dim: usize, n: Option<usize>) -> (FlexTensor, FlexTensor) {
    use burn_std::f16;
    let tensor = super::module::cast_to_f32(tensor, f16::to_f32);
    let (re, im) = rfft_f32(tensor, dim, n);
    (
        super::module::cast_from_f32(re, f16::from_f32),
        super::module::cast_from_f32(im, f16::from_f32),
    )
}

pub fn rfft_bf16(tensor: FlexTensor, dim: usize, n: Option<usize>) -> (FlexTensor, FlexTensor) {
    use burn_std::bf16;
    let tensor = super::module::cast_to_f32(tensor, bf16::to_f32);
    let (re, im) = rfft_f32(tensor, dim, n);
    (
        super::module::cast_from_f32(re, bf16::from_f32),
        super::module::cast_from_f32(im, bf16::from_f32),
    )
}

// ============================================================================
// Inverse real FFT (irfft)
// ============================================================================

/// Inverse complex FFT: IFFT(X) = (1/N) * conj(FFT(conj(X))).
///
/// Conjugates input, runs the forward FFT (with SIMD), conjugates
/// output, and scales by 1/N.
#[inline]
fn inverse_complex_fft(re: &mut [f32], im: &mut [f32], n: usize, tw: &TwiddleRef) {
    if n <= 1 {
        return;
    }

    // Conjugate input
    for v in im.iter_mut() {
        *v = -*v;
    }

    // Forward FFT (mixed radix-4/radix-2, SIMD when available)
    complex_fft(re, im, n, tw);

    // Conjugate output and scale by 1/N
    let scale = 1.0 / n as f32;
    for v in re.iter_mut() {
        *v *= scale;
    }
    for v in im.iter_mut() {
        *v = -*v * scale;
    }
}

/// Repack N/2+1 spectrum bins into N/2 complex values Z[k],
/// reversing rfft's unpack step.
///
/// Z[k] = Xe[k] + i*conj(W)*D[k] where:
///   Xe = (X[k] + conj(X[half-k])) / 2
///   D  = (X[k] - conj(X[half-k])) / 2
///   W  = W_N^k (same twiddle used in rfft unpack)
fn repack_irfft(
    x_re: &[f32],
    x_im: &[f32],
    half: usize,
    tw_re: &[f32],
    tw_im: &[f32],
    z_re: &mut [f32],
    z_im: &mut [f32],
) {
    // k=0: Z[0] = (X[0] + X[half])/2 + i*(X[0] - X[half])/2
    z_re[0] = (x_re[0] + x_re[half]) * 0.5;
    z_im[0] = (x_re[0] - x_re[half]) * 0.5;

    for k in 1..half {
        let j = half - k;
        let (xk_re, xk_im) = (x_re[k], x_im[k]);
        let (xj_re, xj_im) = (x_re[j], x_im[j]);

        // Xe = (X[k] + conj(X[j])) / 2
        let a_re = (xk_re + xj_re) * 0.5;
        let a_im = (xk_im - xj_im) * 0.5;

        // D = (X[k] - conj(X[j])) / 2
        let d_re = (xk_re - xj_re) * 0.5;
        let d_im = (xk_im + xj_im) * 0.5;

        // i*conj(W)*D where W = (wr, wi), conj(W) = (wr, -wi)
        // conj(W)*D = (wr*d_re + wi*d_im, wr*d_im - wi*d_re)
        // i*(...) = (-(wr*d_im - wi*d_re), wr*d_re + wi*d_im)
        let wr = tw_re[k];
        let wi = tw_im[k];

        z_re[k] = a_re - wr * d_im + wi * d_re;
        z_im[k] = a_im + wr * d_re + wi * d_im;
    }
}

/// Process a single irfft fiber via inverse packing trick.
///
/// Repacks N/2+1 spectrum bins into N/2 complex values, runs N/2-point
/// inverse complex FFT, then de-interleaves to N real output values.
#[allow(clippy::too_many_arguments)]
#[inline]
fn irfft_fiber(
    re_in: &[f32],
    im_in: &[f32],
    in_stride: usize,
    half: usize,
    spec_bins: usize,
    signal_out: &mut [f32],
    out_stride: usize,
    tw_half: &TwiddleRef,
    unpack_tw_re: &[f32],
    unpack_tw_im: &[f32],
    z_re: &mut [f32],
    z_im: &mut [f32],
    spec_re: &mut [f32],
    spec_im: &mut [f32],
) {
    if spec_bins > half {
        for k in 0..=half {
            spec_re[k] = re_in[k * in_stride];
            spec_im[k] = im_in[k * in_stride];
        }
    } else {
        for k in 0..=half {
            spec_re[k] = if k < spec_bins {
                re_in[k * in_stride]
            } else {
                0.0
            };
            spec_im[k] = if k < spec_bins {
                im_in[k * in_stride]
            } else {
                0.0
            };
        }
    }

    repack_irfft(
        spec_re,
        spec_im,
        half,
        unpack_tw_re,
        unpack_tw_im,
        z_re,
        z_im,
    );

    // Inverse complex FFT of size N/2
    inverse_complex_fft(z_re, z_im, half, tw_half);

    // De-interleave: signal[2k] = z_re[k], signal[2k+1] = z_im[k]
    if out_stride == 1 {
        for k in 0..half {
            signal_out[2 * k] = z_re[k];
            signal_out[2 * k + 1] = z_im[k];
        }
    } else {
        for k in 0..half {
            signal_out[(2 * k) * out_stride] = z_re[k];
            signal_out[(2 * k + 1) * out_stride] = z_im[k];
        }
    }
}

pub fn irfft_f32(
    spectrum_re: FlexTensor,
    spectrum_im: FlexTensor,
    dim: usize,
    n: Option<usize>,
) -> FlexTensor {
    let spectrum_re = spectrum_re.to_contiguous();
    let spectrum_im = spectrum_im.to_contiguous();
    let shape = spectrum_re.layout().shape().clone();
    assert!(
        *spectrum_im.layout().shape() == shape,
        "irfft: spectrum_re and spectrum_im shapes must match"
    );
    assert!(
        dim < shape.num_dims(),
        "irfft: dim {dim} out of bounds for {}-D tensor",
        shape.num_dims()
    );
    let half_plus_1 = shape[dim];
    assert!(
        half_plus_1 >= 1,
        "irfft: spectrum dimension cannot be empty"
    );

    let spec_bins = half_plus_1;

    let requested_n = n.unwrap_or_else(|| {
        let sig_len = (half_plus_1 - 1) * 2;
        assert!(
            sig_len.is_power_of_two(),
            "irfft: reconstructed signal length must be a power of 2, got {sig_len}"
        );
        sig_len
    });
    let fft_size = requested_n.next_power_of_two();

    // N=1: single DC bin, output is just the real value along `dim`.
    // If caller's spectrum has more bins, take the DC (bin 0) only.
    if fft_size <= 1 {
        let out = if spectrum_re.layout().shape()[dim] != 1 {
            spectrum_re.narrow(dim, 0, 1)
        } else {
            spectrum_re
        };
        return out;
    }

    let half = fft_size / 2;
    let n = fft_size;

    let mut out_dims: Vec<usize> = shape.as_slice().to_vec();
    out_dims[dim] = n;
    let out_shape = Shape::from(out_dims);
    let total_out = out_shape.num_elements();
    let num_fibers = shape.num_elements() / half_plus_1;

    let re_data: &[f32] = spectrum_re.storage();
    let im_data: &[f32] = spectrum_im.storage();
    let in_strides = contiguous_strides_usize(&shape);
    let out_strides = contiguous_strides_usize(&out_shape);

    // Twiddles for N/2-point inverse complex FFT
    let tw_half = get_twiddles(half);

    // Unpack twiddles: last stage of size-N table (same as rfft)
    let tw_full = get_twiddles(n);
    let full_offsets = tw_full.offsets();
    let last_stage_off = if full_offsets.len() >= 2 {
        full_offsets[full_offsets.len() - 2]
    } else {
        0
    };
    let unpack_tw_re = &tw_full.re()[last_stage_off..];
    let unpack_tw_im = &tw_full.im()[last_stage_off..];

    let mut signal_out = vec![0.0f32; total_out];
    let in_stride = in_strides[dim];
    let out_stride = out_strides[dim];

    #[cfg(feature = "rayon")]
    if num_fibers >= 4 && n >= 64 {
        use rayon::prelude::*;

        let fiber_results: Vec<(usize, Vec<f32>)> = (0..num_fibers)
            .into_par_iter()
            .map(|fiber_idx| {
                let re_base = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
                let mut z_re = vec![0.0f32; half.max(1)];
                let mut z_im = vec![0.0f32; half.max(1)];
                let mut spec_re = vec![0.0f32; half + 1];
                let mut spec_im = vec![0.0f32; half + 1];
                let mut fiber_out = vec![0.0f32; n];

                irfft_fiber(
                    &re_data[re_base..],
                    &im_data[re_base..],
                    in_stride,
                    half,
                    spec_bins,
                    &mut fiber_out,
                    1,
                    &tw_half,
                    unpack_tw_re,
                    unpack_tw_im,
                    &mut z_re,
                    &mut z_im,
                    &mut spec_re,
                    &mut spec_im,
                );
                (fiber_idx, fiber_out)
            })
            .collect();

        for (fiber_idx, fiber_out) in fiber_results {
            let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);
            for k in 0..n {
                signal_out[out_base + k * out_stride] = fiber_out[k];
            }
        }

        let result = FlexTensor::new(
            Bytes::from_elems(signal_out),
            Layout::contiguous(out_shape),
            burn_backend::DType::F32,
        );
        return if fft_size > requested_n {
            result.narrow(dim, 0, requested_n)
        } else {
            result
        };
    }

    let mut z_re = vec![0.0f32; half.max(1)];
    let mut z_im = vec![0.0f32; half.max(1)];
    let mut spec_re = vec![0.0f32; half + 1];
    let mut spec_im = vec![0.0f32; half + 1];
    let mut fiber_out = vec![0.0f32; n];

    for fiber_idx in 0..num_fibers {
        let re_base = slice_base_offset(fiber_idx, &shape, &in_strides, dim);
        let out_base = slice_base_offset(fiber_idx, &out_shape, &out_strides, dim);

        irfft_fiber(
            &re_data[re_base..],
            &im_data[re_base..],
            in_stride,
            half,
            spec_bins,
            &mut fiber_out,
            1,
            &tw_half,
            unpack_tw_re,
            unpack_tw_im,
            &mut z_re,
            &mut z_im,
            &mut spec_re,
            &mut spec_im,
        );

        for k in 0..n {
            signal_out[out_base + k * out_stride] = fiber_out[k];
        }
    }

    let result = FlexTensor::new(
        Bytes::from_elems(signal_out),
        Layout::contiguous(out_shape),
        burn_backend::DType::F32,
    );
    if fft_size > requested_n {
        result.narrow(dim, 0, requested_n)
    } else {
        result
    }
}

pub fn irfft_f64(
    spectrum_re: FlexTensor,
    spectrum_im: FlexTensor,
    dim: usize,
    n: Option<usize>,
) -> FlexTensor {
    use burn_backend::DType;
    match spectrum_re.dtype() {
        DType::F64 => {
            let re_f32 = super::module::cast_to_f32::<f64>(spectrum_re, |v| v as f32);
            let im_f32 = super::module::cast_to_f32::<f64>(spectrum_im, |v| v as f32);
            let result = irfft_f32(re_f32, im_f32, dim, n);
            super::module::cast_from_f32::<f64>(result, |v| v as f64)
        }
        _ => irfft_f32(spectrum_re, spectrum_im, dim, n),
    }
}

pub fn irfft_f16(
    spectrum_re: FlexTensor,
    spectrum_im: FlexTensor,
    dim: usize,
    n: Option<usize>,
) -> FlexTensor {
    use burn_std::f16;
    let re = super::module::cast_to_f32(spectrum_re, f16::to_f32);
    let im = super::module::cast_to_f32(spectrum_im, f16::to_f32);
    let result = irfft_f32(re, im, dim, n);
    super::module::cast_from_f32(result, f16::from_f32)
}

pub fn irfft_bf16(
    spectrum_re: FlexTensor,
    spectrum_im: FlexTensor,
    dim: usize,
    n: Option<usize>,
) -> FlexTensor {
    use burn_std::bf16;
    let re = super::module::cast_to_f32(spectrum_re, bf16::to_f32);
    let im = super::module::cast_to_f32(spectrum_im, bf16::to_f32);
    let result = irfft_f32(re, im, dim, n);
    super::module::cast_from_f32(result, bf16::from_f32)
}

// Tests kept here exercise flex-specific internals: the FFT kernels
// (`rfft_f32`/`_f64`/`_f16`, `irfft_*`, `complex_fft`, `inverse_complex_fft`)
// across sizes that span the radix-4 and complex packing paths (N=1, 2, 4, 8,
// 256, 1024, 4096), f16/f64 dtype handling, twiddle accuracy, Parseval's
// theorem on synthetic inputs, and a reference cross-check against realfft.
#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{DType, TensorData, Tolerance};

    fn make_f32(data: Vec<f32>, shape: Vec<usize>) -> FlexTensor {
        FlexTensor::from_data(TensorData::new(data, shape))
    }

    fn make_f64(data: Vec<f64>, shape: Vec<usize>) -> FlexTensor {
        FlexTensor::from_data(TensorData::new(data, shape))
    }

    fn assert_approx(tensor: FlexTensor, expected: &[f32], tol: f32) {
        let shape = tensor.layout().shape().as_slice().to_vec();
        tensor.into_data().assert_approx_eq::<f32>(
            &TensorData::new(expected.to_vec(), shape),
            Tolerance::absolute(tol),
        );
    }

    fn assert_approx_f64(tensor: FlexTensor, expected: &[f64], tol: f64) {
        tensor
            .into_data()
            .assert_approx_eq::<f64>(&TensorData::from(expected), Tolerance::absolute(tol));
    }

    // ---- N=1 ----

    #[test]
    fn rfft_n1() {
        let signal = make_f32(vec![5.0], vec![1]);
        let (re, im) = rfft_f32(signal, 0, None);
        assert_approx(re, &[5.0], 1e-6);
        assert_approx(im, &[0.0], 1e-6);
    }

    // ---- N=2 ----

    #[test]
    fn rfft_n2() {
        let signal = make_f32(vec![1.0, -1.0], vec![2]);
        let (re, im) = rfft_f32(signal, 0, None);
        assert_approx(re, &[0.0, 2.0], 1e-6);
        assert_approx(im, &[0.0, 0.0], 1e-6);
    }

    // ---- N=4: known DFT of [1,0,0,0] = [1,1,1] (all real) ----

    #[test]
    fn rfft_n4_impulse() {
        let signal = make_f32(vec![1.0, 0.0, 0.0, 0.0], vec![4]);
        let (re, im) = rfft_f32(signal, 0, None);
        assert_approx(re, &[1.0, 1.0, 1.0], 1e-6);
        assert_approx(im, &[0.0, 0.0, 0.0], 1e-6);
    }

    // ---- N=4: constant signal [1,1,1,1] -> DC only ----

    #[test]
    fn rfft_n4_constant() {
        let signal = make_f32(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let (re, im) = rfft_f32(signal, 0, None);
        assert_approx(re, &[4.0, 0.0, 0.0], 1e-6);
        assert_approx(im, &[0.0, 0.0, 0.0], 1e-6);
    }

    // ---- N=4: zeros ----

    #[test]
    fn rfft_n4_zeros() {
        let signal = make_f32(vec![0.0; 4], vec![4]);
        let (re, im) = rfft_f32(signal, 0, None);
        assert_approx(re, &[0.0, 0.0, 0.0], 1e-6);
        assert_approx(im, &[0.0, 0.0, 0.0], 1e-6);
    }

    // ---- N=8 ----

    #[test]
    fn rfft_n8_impulse() {
        let mut signal = vec![0.0f32; 8];
        signal[0] = 1.0;
        let (re, im) = rfft_f32(make_f32(signal, vec![8]), 0, None);
        // DFT of impulse is all 1s
        assert_approx(re, &[1.0, 1.0, 1.0, 1.0, 1.0], 1e-6);
        assert_approx(im, &[0.0, 0.0, 0.0, 0.0, 0.0], 1e-6);
    }

    #[test]
    fn rfft_n8_cosine() {
        // cos(2*pi*k/8) for k=0..7 -> energy at bin 1
        let signal: Vec<f32> = (0..8)
            .map(|k| (2.0 * std::f32::consts::PI * k as f32 / 8.0).cos())
            .collect();
        let (re, im) = rfft_f32(make_f32(signal, vec![8]), 0, None);
        // Bin 1 should have amplitude 4 (real), rest ~0
        assert_approx(re, &[0.0, 4.0, 0.0, 0.0, 0.0], 1e-4);
        assert_approx(im, &[0.0, 0.0, 0.0, 0.0, 0.0], 1e-4);
    }

    // ---- Larger size: N=256 ----

    #[test]
    fn rfft_n256_impulse() {
        let mut signal = vec![0.0f32; 256];
        signal[0] = 1.0;
        let (re, im) = rfft_f32(make_f32(signal, vec![256]), 0, None);
        let re_data = re.into_data();
        let im_data = im.into_data();
        let re_vals = re_data.as_slice::<f32>().unwrap();
        let im_vals = im_data.as_slice::<f32>().unwrap();
        assert_eq!(re_vals.len(), 129);
        for &v in re_vals {
            assert!((v - 1.0).abs() < 1e-5, "re bin should be 1.0, got {v}");
        }
        for &v in im_vals {
            assert!(v.abs() < 1e-5, "im bin should be 0.0, got {v}");
        }
    }

    // ---- Multi-dimensional: FFT along dim 1 ----

    #[test]
    fn rfft_2d_dim1() {
        // 2 rows, each of length 4: impulse and constant
        let data = vec![
            1.0, 0.0, 0.0, 0.0, // row 0: impulse
            1.0, 1.0, 1.0, 1.0, // row 1: constant
        ];
        let signal = make_f32(data, vec![2, 4]);
        let (re, im) = rfft_f32(signal, 1, None);
        // Shape should be [2, 3]
        let re_data = re.into_data();
        let im_data = im.into_data();
        let re_vals = re_data.as_slice::<f32>().unwrap();
        let im_vals = im_data.as_slice::<f32>().unwrap();
        assert_eq!(re_vals.len(), 6); // 2 * 3
        // Row 0 (impulse): [1, 1, 1]
        assert!((re_vals[0] - 1.0).abs() < 1e-5);
        assert!((re_vals[1] - 1.0).abs() < 1e-5);
        assert!((re_vals[2] - 1.0).abs() < 1e-5);
        // Row 1 (constant): [4, 0, 0]
        assert!((re_vals[3] - 4.0).abs() < 1e-5);
        assert!((re_vals[4]).abs() < 1e-5);
        assert!((re_vals[5]).abs() < 1e-5);
        // All imaginary should be ~0
        for &v in im_vals {
            assert!(v.abs() < 1e-5);
        }
    }

    // ---- FFT along dim 0 ----

    #[test]
    fn rfft_2d_dim0() {
        // 4 rows, 2 cols: impulse in each column
        let data = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let signal = make_f32(data, vec![4, 2]);
        let (re, _im) = rfft_f32(signal, 0, None);
        // Shape should be [3, 2]
        let re_data = re.into_data();
        let re_vals = re_data.as_slice::<f32>().unwrap();
        assert_eq!(re_vals.len(), 6);
        // Each column is an impulse -> all bins = 1
        for &v in re_vals {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    // ---- f64 dtype ----

    #[test]
    fn rfft_f64_n4_impulse() {
        let signal = make_f64(vec![1.0, 0.0, 0.0, 0.0], vec![4]);
        let (re, im) = rfft_f64(signal, 0, None);
        assert_approx_f64(re, &[1.0, 1.0, 1.0], 1e-10);
        assert_approx_f64(im, &[0.0, 0.0, 0.0], 1e-10);
    }

    #[test]
    fn rfft_f64_n8_cosine() {
        let signal: Vec<f64> = (0..8)
            .map(|k| (2.0 * std::f64::consts::PI * k as f64 / 8.0).cos())
            .collect();
        let (re, im) = rfft_f64(make_f64(signal, vec![8]), 0, None);
        assert_approx_f64(re, &[0.0, 4.0, 0.0, 0.0, 0.0], 1e-6);
        assert_approx_f64(im, &[0.0, 0.0, 0.0, 0.0, 0.0], 1e-6);
    }

    // ---- f16 dtype ----

    #[test]
    fn rfft_f16_n4_impulse() {
        use burn_std::f16;
        let f16_data = vec![
            f16::from_f32(1.0),
            f16::from_f32(0.0),
            f16::from_f32(0.0),
            f16::from_f32(0.0),
        ];
        let signal = FlexTensor::new(
            Bytes::from_elems(f16_data),
            Layout::contiguous(Shape::from(vec![4])),
            DType::F16,
        );
        let (re, _im) = rfft_f16(signal, 0, None);
        // Verify via round-trip to f32
        let re_f32 = super::super::module::cast_to_f32(re, f16::to_f32);
        let re_data = re_f32.into_data();
        let re_vals = re_data.as_slice::<f32>().unwrap();
        assert_eq!(re_vals.len(), 3);
        for &v in re_vals {
            assert!((v - 1.0).abs() < 0.01, "expected ~1.0, got {v}");
        }
    }

    // ---- Const twiddle accuracy ----

    #[test]
    fn const_sin_cos_accuracy() {
        let test_angles = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, -1.0, -3.0, 6.0];
        for &angle in &test_angles {
            let cs = const_sin(angle);
            let cc = const_cos(angle);
            let rs = angle.sin();
            let rc = angle.cos();
            assert!(
                (cs - rs).abs() < 1e-12,
                "const_sin({angle}) = {cs}, expected {rs}"
            );
            assert!(
                (cc - rc).abs() < 1e-12,
                "const_cos({angle}) = {cc}, expected {rc}"
            );
        }
    }

    // ---- N=1024 round-trip with known property: Parseval's theorem ----
    // Sum of |x|^2 = (1/N) * Sum of |X|^2

    #[test]
    fn rfft_n1024_parseval() {
        let n = 1024;
        let signal: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let time_energy: f64 = signal.iter().map(|&x| (x as f64) * (x as f64)).sum();

        let (re, im) = rfft_f32(make_f32(signal, vec![n]), 0, None);
        let re_data = re.into_data();
        let im_data = im.into_data();
        let re_vals = re_data.as_slice::<f32>().unwrap();
        let im_vals = im_data.as_slice::<f32>().unwrap();

        // Frequency energy: DC and Nyquist count once, others count double
        let out_len = n / 2 + 1;
        let mut freq_energy = 0.0f64;
        for k in 0..out_len {
            let mag2 = (re_vals[k] as f64).powi(2) + (im_vals[k] as f64).powi(2);
            if k == 0 || k == n / 2 {
                freq_energy += mag2;
            } else {
                freq_energy += 2.0 * mag2;
            }
        }
        freq_energy /= n as f64;

        let rel_err = (freq_energy - time_energy).abs() / time_energy;
        assert!(
            rel_err < 1e-4,
            "Parseval's theorem violated: time={time_energy}, freq={freq_energy}, rel_err={rel_err}"
        );
    }

    // ---- irfft tests ----

    #[test]
    fn irfft_roundtrip_n4() {
        let signal = make_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let (re, im) = rfft_f32(signal.clone(), 0, None);
        let reconstructed = irfft_f32(re, im, 0, None);
        assert_approx(reconstructed, &[1.0, 2.0, 3.0, 4.0], 1e-5);
    }

    #[test]
    fn irfft_roundtrip_n8() {
        let data: Vec<f32> = (0..8).map(|i| (i as f32 * 0.3).sin()).collect();
        let signal = make_f32(data.clone(), vec![8]);
        let (re, im) = rfft_f32(signal, 0, None);
        let reconstructed = irfft_f32(re, im, 0, None);
        assert_approx(reconstructed, &data, 1e-5);
    }

    #[test]
    fn rfft_vs_realfft() {
        // Verify our rfft matches realfft (rustfft-backed) for non-trivial input
        // at sizes that exercise radix-4 (n>=16) and the complex packing trick.
        use realfft::RealFftPlanner;

        let mut planner = RealFftPlanner::<f32>::new();

        for &n in &[4, 8, 16, 32, 64, 256, 1024, 4096] {
            let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() + 0.5).collect();

            // Our rfft
            let signal = make_f32(data.clone(), vec![n]);
            let (re_out, im_out) = rfft_f32(signal, 0, None);
            let re_data = re_out.into_data();
            let im_data = im_out.into_data();
            let our_re = re_data.as_slice::<f32>().unwrap();
            let our_im = im_data.as_slice::<f32>().unwrap();

            // Reference: realfft
            let r2c = planner.plan_fft_forward(n);
            let mut input = data.clone();
            let mut spectrum = r2c.make_output_vec();
            r2c.process(&mut input, &mut spectrum).unwrap();

            let out_len = n / 2 + 1;
            assert_eq!(our_re.len(), out_len);
            assert_eq!(spectrum.len(), out_len);

            let max_re_err = our_re
                .iter()
                .zip(spectrum.iter())
                .map(|(&a, b)| (a - b.re).abs())
                .fold(0.0f32, f32::max);
            let max_im_err = our_im
                .iter()
                .zip(spectrum.iter())
                .map(|(&a, b)| (a - b.im).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_re_err < 1e-3 && max_im_err < 1e-3,
                "rfft vs realfft mismatch at n={n}: max_re_err={max_re_err}, max_im_err={max_im_err}"
            );
        }
    }

    #[test]
    fn irfft_vs_realfft() {
        // Verify our irfft matches realfft's inverse for non-trivial spectra.
        use realfft::RealFftPlanner;

        let mut planner = RealFftPlanner::<f32>::new();

        for &n in &[4, 8, 16, 32, 64, 256, 1024, 4096] {
            // Generate a spectrum via realfft forward
            let r2c = planner.plan_fft_forward(n);
            let c2r = planner.plan_fft_inverse(n);
            let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() + 0.5).collect();
            let mut input = data.clone();
            let mut spectrum = r2c.make_output_vec();
            r2c.process(&mut input, &mut spectrum).unwrap();

            // Our irfft
            let out_len = n / 2 + 1;
            let spec_re: Vec<f32> = spectrum.iter().map(|c| c.re).collect();
            let spec_im: Vec<f32> = spectrum.iter().map(|c| c.im).collect();
            let re_tensor = make_f32(spec_re, vec![out_len]);
            let im_tensor = make_f32(spec_im, vec![out_len]);
            let our_result = irfft_f32(re_tensor, im_tensor, 0, None);
            let our_data = our_result.into_data();
            let our_vals = our_data.as_slice::<f32>().unwrap();

            // Reference: realfft inverse (note: realfft doesn't normalize, so scale)
            let mut spec_copy = spectrum.clone();
            let mut ref_output = c2r.make_output_vec();
            c2r.process(&mut spec_copy, &mut ref_output).unwrap();
            let scale = 1.0 / n as f32;
            let ref_scaled: Vec<f32> = ref_output.iter().map(|&v| v * scale).collect();

            let max_err = our_vals
                .iter()
                .zip(ref_scaled.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 1e-3,
                "irfft vs realfft mismatch at n={n}: max_err={max_err}"
            );
        }
    }

    #[test]
    fn forward_complex_fft_impulse() {
        // DFT of impulse [1,0,0,...,0] should be all 1s
        for &n in &[4, 8, 16, 32, 64] {
            let tw = get_twiddles(n);
            let mut re = vec![0.0f32; n];
            let mut im = vec![0.0f32; n];
            re[0] = 1.0;
            complex_fft(&mut re, &mut im, n, &tw);
            let max_re_err = re.iter().map(|&v| (v - 1.0).abs()).fold(0.0f32, f32::max);
            let max_im_err = im.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
            assert!(
                max_re_err < 1e-5 && max_im_err < 1e-5,
                "forward FFT impulse n={n}: max_re_err={max_re_err}, max_im_err={max_im_err}"
            );
        }
    }

    #[test]
    fn inverse_complex_fft_roundtrip() {
        for &n in &[4, 8, 16, 32, 64, 256] {
            let tw = get_twiddles(n);
            let mut re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
            let mut im = vec![0.0f32; n];
            let orig_re = re.clone();

            complex_fft(&mut re, &mut im, n, &tw);
            inverse_complex_fft(&mut re, &mut im, n, &tw);

            let max_err = re
                .iter()
                .zip(orig_re.iter())
                .map(|(&got, &expected)| (got - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 1e-4,
                "inverse_complex_fft roundtrip n={n}: max error {max_err}"
            );
        }
    }

    #[test]
    fn irfft_roundtrip_n256() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).cos()).collect();
        let signal = make_f32(data.clone(), vec![256]);
        let (re, im) = rfft_f32(signal, 0, None);
        let reconstructed = irfft_f32(re, im, 0, None);
        let result = reconstructed.into_data();
        let vals = result.as_slice::<f32>().unwrap();
        let max_err = vals
            .iter()
            .zip(data.iter())
            .map(|(&got, &expected)| (got - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 5e-3,
            "irfft_roundtrip_n256: max error {max_err} exceeds tolerance"
        );
    }

    #[test]
    fn irfft_roundtrip_2d_dim1() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, // row 0
            5.0, 6.0, 7.0, 8.0, // row 1
        ];
        let signal = make_f32(data.clone(), vec![2, 4]);
        let (re, im) = rfft_f32(signal, 1, None);
        let reconstructed = irfft_f32(re, im, 1, None);
        assert_approx(reconstructed, &data, 1e-5);
    }

    #[test]
    fn irfft_roundtrip_2d_dim0() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let signal = make_f32(data.clone(), vec![4, 2]);
        let (re, im) = rfft_f32(signal, 0, None);
        let reconstructed = irfft_f32(re, im, 0, None);
        assert_approx(reconstructed, &data, 1e-5);
    }

    #[test]
    fn irfft_known_spectrum() {
        // DC=4, all others zero -> constant signal [1,1,1,1]
        let re = make_f32(vec![4.0, 0.0, 0.0], vec![3]);
        let im = make_f32(vec![0.0, 0.0, 0.0], vec![3]);
        let signal = irfft_f32(re, im, 0, None);
        assert_approx(signal, &[1.0, 1.0, 1.0, 1.0], 1e-5);
    }

    #[test]
    fn irfft_f64_roundtrip() {
        // irfft_f64 truncates to f32 internally, so tolerance is f32-level
        let data: Vec<f64> = (0..8).map(|i| (i as f64 * 0.3).sin()).collect();
        let signal = make_f64(data.clone(), vec![8]);
        let (re, im) = rfft_f64(signal, 0, None);
        let reconstructed = irfft_f64(re, im, 0, None);
        assert_approx_f64(reconstructed, &data, 1e-5);
    }

    // Coverage for the n=Some(pow2) path on flex.

    #[test]
    fn rfft_n_larger_than_signal_zero_pads() {
        // n=8 zero-pads the signal; output has 8/2+1 = 5 bins.
        let signal = make_f32(vec![1.0, 0.0, 0.0, 0.0], vec![4]);
        let (re, im) = rfft_f32(signal, 0, Some(8));
        assert_eq!(re.layout().shape()[0], 5);
        assert_eq!(im.layout().shape()[0], 5);
        let re_vals = re.into_data().as_slice::<f32>().unwrap().to_vec();
        for (k, v) in re_vals.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "impulse DFT re[{k}] should be 1.0, got {v}"
            );
        }
    }

    #[test]
    fn rfft_n_smaller_than_signal_truncates_first() {
        // Signal length 8, n=4 -> truncate to 4, compute 4-point DFT of [1,0,0,0].
        let signal = make_f32(vec![1.0, 0.0, 0.0, 0.0, 99.0, 99.0, 99.0, 99.0], vec![8]);
        let (re, _im) = rfft_f32(signal, 0, Some(4));
        assert_eq!(re.layout().shape()[0], 3);
        let re_vals = re.into_data().as_slice::<f32>().unwrap().to_vec();
        for v in &re_vals {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn rfft_irfft_roundtrip_with_pow2_n() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let signal = make_f32(data.clone(), vec![8]);
        let (re, im) = rfft_f32(signal, 0, Some(8));
        let reconstructed = irfft_f32(re, im, 0, Some(8));
        assert_approx(reconstructed, &data, 1e-4);
    }

    #[test]
    fn rfft_f64_with_pow2_n_and_truncation() {
        // Signal length 8, n=4. Output has 4/2+1 = 3 bins.
        let data: Vec<f64> = (0..8).map(|i| (i as f64 * 0.3).sin()).collect();
        let signal = make_f64(data, vec![8]);
        let (re, im) = rfft_f64(signal, 0, Some(4));
        assert_eq!(re.layout().shape()[0], 3);
        assert_eq!(im.layout().shape()[0], 3);
    }

    #[test]
    fn rfft_vs_realfft_with_pow2_n_and_padding() {
        use realfft::RealFftPlanner;

        let mut planner = RealFftPlanner::<f32>::new();
        // Signal shorter than n (pow2); backend zero-pads before the FFT.
        for &(sig_len, n) in &[(3usize, 4usize), (5, 8), (6, 8), (9, 16)] {
            let data: Vec<f32> = (0..sig_len)
                .map(|i| (i as f32 * 0.41).cos() - 0.2)
                .collect();
            let mut padded = data.clone();
            padded.resize(n, 0.0);

            let r2c = planner.plan_fft_forward(n);
            let mut input = padded;
            let mut ref_spec = r2c.make_output_vec();
            r2c.process(&mut input, &mut ref_spec).unwrap();

            let signal = make_f32(data, vec![sig_len]);
            let (re, im) = rfft_f32(signal, 0, Some(n));
            let re_v = re.into_data().as_slice::<f32>().unwrap().to_vec();
            let im_v = im.into_data().as_slice::<f32>().unwrap().to_vec();

            assert_eq!(re_v.len(), n / 2 + 1);
            for (k, refc) in ref_spec.iter().enumerate() {
                let err_re = (re_v[k] - refc.re).abs();
                let err_im = (im_v[k] - refc.im).abs();
                assert!(
                    err_re < 1e-3 && err_im < 1e-3,
                    "sig_len={sig_len} n={n} bin={k}: got ({}, {}), ref ({}, {})",
                    re_v[k],
                    im_v[k],
                    refc.re,
                    refc.im
                );
            }
        }
    }
}
