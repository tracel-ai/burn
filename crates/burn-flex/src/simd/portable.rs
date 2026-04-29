//! Portable SIMD kernels using macerator.
//!
//! Replaces platform-specific implementations (neon.rs) with a single
//! portable implementation that auto-dispatches to NEON/AVX2/SSE/SIMD128/scalar.

use macerator::{Scalar, Simd, VBitAnd, VBitOr, VBitXor, vload_unaligned, vstore_unaligned};

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

macro_rules! define_inplace_f32_op {
    ($pub_fn:ident, $seq_fn:ident, $par_fn:ident, $op:tt) => {
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
                    let va = vload_unaligned(a.as_ptr().add(i));
                    let vb = vload_unaligned(b.as_ptr().add(i));
                    vstore_unaligned::<S, _>(a.as_mut_ptr().add(i), va $op vb);
                }
                i += lanes;
            }

            for j in simd_len..len {
                a[j] = a[j] $op b[j];
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

define_inplace_f32_op!(add_inplace_f32, add_inplace_f32_seq, add_inplace_f32_par, +);
define_inplace_f32_op!(sub_inplace_f32, sub_inplace_f32_seq, sub_inplace_f32_par, -);
define_inplace_f32_op!(mul_inplace_f32, mul_inplace_f32_seq, mul_inplace_f32_par, *);
define_inplace_f32_op!(div_inplace_f32, div_inplace_f32_seq, div_inplace_f32_par, /);

// ============================================================================
// f32 shared-row broadcast binary ops
// ============================================================================
//
// Pattern: `dst[r * row_len + j] = dst[r * row_len + j] op row[j]` for
// all `r in 0..num_rows, j in 0..row_len`. That is, a single contiguous
// `row` is broadcast and combined with each of `num_rows` consecutive
// rows in `dst`. This is what `gamma.unsqueeze() * x` and friends
// produce after broadcast expansion.
//
// We call the SIMD dispatch once for the entire walk so that
// `#[macerator::with_simd]`'s feature detection pays once, not once per
// row. Calling the normal `add_inplace_f32` in a `0..num_rows` loop
// gives the right answer but costs ~55k dispatches on the typical
// layer_norm shape; see issue #64 item 2 benchmarks.

macro_rules! define_shared_row_f32_op {
    ($pub_fn:ident, $seq_fn:ident, $par_fn:ident, $op:tt) => {
        #[inline]
        pub fn $pub_fn(dst: &mut [f32], row: &[f32]) {
            let row_len = row.len();
            if row_len == 0 || dst.is_empty() {
                return;
            }
            debug_assert_eq!(
                dst.len() % row_len,
                0,
                "shared-row broadcast: dst length must be a multiple of row length"
            );

            #[cfg(feature = "rayon")]
            if dst.len() >= PARALLEL_THRESHOLD {
                $par_fn(dst, row);
                return;
            }

            $seq_fn(dst, row);
        }

        #[macerator::with_simd]
        #[allow(clippy::assign_op_pattern)]
        fn $seq_fn<S: Simd>(dst: &mut [f32], row: &[f32]) {
            let lanes = S::lanes32();
            let row_len = row.len();
            let simd_len = row_len / lanes * lanes;
            let num_rows = dst.len() / row_len;

            for r in 0..num_rows {
                let base = r * row_len;
                let mut i = 0;
                while i < simd_len {
                    unsafe {
                        let va = vload_unaligned::<S, _>(dst.as_ptr().add(base + i));
                        let vb = vload_unaligned::<S, _>(row.as_ptr().add(i));
                        vstore_unaligned::<S, _>(
                            dst.as_mut_ptr().add(base + i),
                            va $op vb,
                        );
                    }
                    i += lanes;
                }
                for j in simd_len..row_len {
                    dst[base + j] = dst[base + j] $op row[j];
                }
            }
        }

        #[cfg(feature = "rayon")]
        fn $par_fn(dst: &mut [f32], row: &[f32]) {
            let row_len = row.len();
            // Chunk on row boundaries so each worker sees whole rows.
            let rows_per_chunk = (CHUNK_SIZE / row_len).max(1);
            let chunk_elems = rows_per_chunk * row_len;
            dst.par_chunks_mut(chunk_elems).for_each(|chunk| {
                $seq_fn(chunk, row);
            });
        }
    };
}

define_shared_row_f32_op!(
    add_shared_row_inplace_f32,
    add_shared_row_inplace_f32_seq,
    add_shared_row_inplace_f32_par,
    +
);
define_shared_row_f32_op!(
    sub_shared_row_inplace_f32,
    sub_shared_row_inplace_f32_seq,
    sub_shared_row_inplace_f32_par,
    -
);
define_shared_row_f32_op!(
    mul_shared_row_inplace_f32,
    mul_shared_row_inplace_f32_seq,
    mul_shared_row_inplace_f32_par,
    *
);
define_shared_row_f32_op!(
    div_shared_row_inplace_f32,
    div_shared_row_inplace_f32_seq,
    div_shared_row_inplace_f32_par,
    /
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
        let value: Vec<f64> = (0..n).map(|i| (i as f64) * -1.0).collect();
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
