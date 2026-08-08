//! Binary tensor operations (add, sub, mul, div).

use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, bf16, f16};

use crate::FlexTensor;
use crate::layout::Layout;
use crate::strided_index::StridedIter;

#[cfg(feature = "simd")]
use crate::simd;

/// Operation type for SIMD dispatch.
#[derive(Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Apply a binary operation element-wise to two tensors.
///
/// Requires tensors to have the same shape. Uses SIMD acceleration for f32
/// when available and both tensors are contiguous.
///
/// Pass `simd_hint` to enable direct SIMD dispatch for standard ops (add/sub/mul/div).
/// Pass `None` for custom operations that have no SIMD fast path.
pub fn binary_op<F32Op, F64Op>(
    lhs: FlexTensor,
    rhs: FlexTensor,
    f32_op: F32Op,
    f64_op: F64Op,
    simd_hint: Option<BinaryOp>,
) -> FlexTensor
where
    F32Op: Fn(f32, f32) -> f32 + Copy,
    F64Op: Fn(f64, f64) -> f64 + Copy,
{
    debug_assert_eq!(lhs.dtype(), rhs.dtype(), "binary_op: dtype mismatch");

    // Broadcast tensors to the same shape if needed
    let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

    let dtype = lhs.dtype();

    match dtype {
        DType::F32 => binary_op_f32(lhs, rhs, f32_op, simd_hint),
        DType::F64 => binary_op_typed(lhs, &rhs, f64_op),
        DType::F16 => binary_op_typed(lhs, &rhs, |a: f16, b: f16| {
            f16::from_f32(f32_op(a.to_f32(), b.to_f32()))
        }),
        DType::BF16 => binary_op_typed(lhs, &rhs, |a: bf16, b: bf16| {
            bf16::from_f32(f32_op(a.to_f32(), b.to_f32()))
        }),
        _ => panic!("binary_op: unsupported dtype {:?}", dtype),
    }
}

/// Specialized binary operation for f32 with SIMD fast path.
#[cfg(feature = "simd")]
fn binary_op_f32<Op>(
    mut lhs: FlexTensor,
    mut rhs: FlexTensor,
    op: Op,
    simd_hint: Option<BinaryOp>,
) -> FlexTensor
where
    Op: Fn(f32, f32) -> f32,
{
    // Permuted lhs + broadcast rhs (e.g. `x.permute(...) - mean`):
    // the generic path would walk the permuted lhs with a scalar
    // `StridedIter`, an order of magnitude slower than the SIMD fast
    // path. Pay one memcpy to materialize lhs contiguous so the fast
    // paths below can take over.
    //
    // Gate on `simd_hint.is_some()` so custom ops like `atan2` or
    // `powf` (which have no SIMD fast path and go straight to
    // `binary_op_typed`) don't pay for a memcpy they can't benefit
    // from. Their strided fallback handles non-contig lhs directly.
    if simd_hint.is_some() && !lhs.layout().is_contiguous() && rhs.layout().strides().contains(&0) {
        lhs = lhs.to_contiguous();
    }

    // Mirror of the above for the swapped orientation: broadcast lhs +
    // permuted rhs (e.g. `mean - x.permute(...)`). Materialize rhs so
    // the swapped broadcast fast path below can take over.
    if simd_hint.is_some() && !rhs.layout().is_contiguous() && lhs.layout().strides().contains(&0) {
        rhs = rhs.to_contiguous();
    }

    // In-place SIMD fast path: lhs unique contiguous at offset 0, rhs
    // contiguous (no broadcast).
    if let Some(simd_op) = simd_hint
        && lhs.is_unique()
        && let (Some((0, l_end)), Some((r_start, r_end))) = (
            lhs.layout().contiguous_offsets(),
            rhs.layout().contiguous_offsets(),
        )
    {
        let r_slice: &[f32] = &rhs.storage()[r_start..r_end];
        let lhs_storage: &mut [f32] = lhs.storage_mut();
        let l_slice = &mut lhs_storage[..l_end];

        match simd_op {
            BinaryOp::Add => simd::add_inplace_f32(l_slice, r_slice),
            BinaryOp::Sub => simd::sub_inplace_f32(l_slice, r_slice),
            BinaryOp::Mul => simd::mul_inplace_f32(l_slice, r_slice),
            BinaryOp::Div => simd::div_inplace_f32(l_slice, r_slice),
        }
        return lhs;
    }

    // Swapped in-place SIMD fast path: lhs is contiguous but not
    // eligible as the destination (shared, or offset != 0), while rhs
    // is unique contiguous at offset 0. Write into rhs instead. Add and
    // mul commute; sub and div use the reversed kernels, which compute
    // `src OP dst` so the result is still `lhs OP rhs`.
    if let Some(simd_op) = simd_hint
        && rhs.is_unique()
        && let (Some((l_start, l_end)), Some((0, r_end))) = (
            lhs.layout().contiguous_offsets(),
            rhs.layout().contiguous_offsets(),
        )
    {
        let l_slice: &[f32] = &lhs.storage()[l_start..l_end];
        let rhs_storage: &mut [f32] = rhs.storage_mut();
        let r_slice = &mut rhs_storage[..r_end];

        match simd_op {
            BinaryOp::Add => simd::add_inplace_f32(r_slice, l_slice),
            BinaryOp::Sub => simd::rsub_inplace_f32(r_slice, l_slice),
            BinaryOp::Mul => simd::mul_inplace_f32(r_slice, l_slice),
            BinaryOp::Div => simd::rdiv_inplace_f32(r_slice, l_slice),
        }
        return rhs;
    }

    // Broadcast SIMD fast path: rhs is broadcast via stride-0 dims in
    // one of two hot shapes that dominate layer_norm decomposition --
    // shared-row (`gamma.unsqueeze() * x`) or per-row scalar
    // (`x - x.mean_dim(-1)`).
    if let Some(simd_op) = simd_hint
        && let Some(pattern) = detect_broadcast_pattern(lhs.layout(), &rhs)
    {
        return apply_broadcast_pattern_f32(lhs, &rhs, simd_op, pattern, false);
    }

    // Swapped broadcast SIMD fast path: the *lhs* is the broadcast
    // operand and rhs is the dense one (e.g. `[1,1,N] * [B,S,N]` from
    // `gamma.unsqueeze() * x`). Run the same pattern detection with the
    // roles reversed, dispatching to reversed kernels for the
    // non-commutative ops so the result is still `lhs OP rhs`.
    if let Some(simd_op) = simd_hint
        && let Some(pattern) = detect_broadcast_pattern(rhs.layout(), &lhs)
    {
        return apply_broadcast_pattern_f32(rhs, &lhs, simd_op, pattern, true);
    }

    binary_op_typed(lhs, &rhs, op)
}

/// Categorization of the two broadcast patterns we can accelerate.
///
/// The two operands are the *dense* one (row-contiguous at offset 0,
/// used as the destination) and the *broadcast* one (`bcast`, the
/// operand with stride-0 dims). In the natural orientation dense=lhs
/// and bcast=rhs; in the swapped orientation the roles are reversed
/// and the non-commutative ops dispatch to reversed kernels.
#[cfg(feature = "simd")]
#[derive(Debug, Clone, Copy)]
enum BroadcastView {
    /// bcast's inner `row_len` elements form a contiguous row that is
    /// shared across `outer_count` outer positions. Starts at
    /// `bcast_row_offset` in bcast's storage.
    SharedRow {
        outer_count: usize,
        row_len: usize,
        bcast_row_offset: usize,
    },
    /// bcast's inner `row_len` elements are all the same scalar,
    /// stepping through `outer_count` scalars along the outer dims
    /// starting at `bcast_scalar_base` in bcast's storage.
    PerRowScalar {
        outer_count: usize,
        row_len: usize,
        bcast_scalar_base: usize,
    },
}

/// Detect whether the broadcast operand can be handled as one of the
/// accelerated broadcast patterns, returning `None` if the stride
/// pattern doesn't fit either bucket or the resulting offsets would
/// leave bcast's storage.
#[cfg(feature = "simd")]
fn detect_broadcast_pattern(dense: &Layout, bcast: &FlexTensor) -> Option<BroadcastView> {
    let bcast_layout = bcast.layout();
    let bcast_storage_elems = bcast.storage::<f32>().len();
    // Require the dense operand to be row-contiguous at offset 0. The
    // broadcast kernel below uses linear offsets into dense's storage;
    // relaxing this would complicate the indexing without helping the
    // hot layer_norm path.
    let (dense_start, _) = dense.contiguous_offsets()?;
    if dense_start != 0 {
        return None;
    }
    let ndims = dense.num_dims();
    if ndims == 0 || bcast_layout.num_dims() != ndims {
        return None;
    }
    let dense_shape = dense.shape();
    let bcast_strides = bcast_layout.strides();

    let last_stride = bcast_strides[ndims - 1];

    // Case A: shared row. Innermost bcast stride is 1, and every outer
    // dim either has stride 0 (a broadcast dim) or size 1 (stride
    // doesn't matter since the dim never advances past index 0).
    if last_stride == 1 {
        let outer_ok = (0..ndims - 1).all(|d| bcast_strides[d] == 0 || dense_shape[d] == 1);
        if outer_ok {
            let outer_count: usize = (0..ndims - 1).map(|d| dense_shape[d]).product();
            let row_len = dense_shape[ndims - 1];
            if outer_count == 0 || row_len == 0 {
                return None;
            }
            let bcast_row_offset = bcast_layout.start_offset();
            // Bounds: kernel reads `bcast_storage[off..off+row_len]`.
            if bcast_row_offset.checked_add(row_len)? > bcast_storage_elems {
                return None;
            }
            return Some(BroadcastView::SharedRow {
                outer_count,
                row_len,
                bcast_row_offset,
            });
        }
    }

    // Case B: per-row scalar. Innermost dims all have stride 0 in
    // bcast and outer dims walk bcast contiguously in row-major order.
    if last_stride == 0 {
        // Count the trailing stride-0 dims to find the inner scalar span.
        let mut inner_dims = 0usize;
        let mut row_len: usize = 1;
        for d in (0..ndims).rev() {
            if bcast_strides[d] == 0 {
                inner_dims += 1;
                row_len *= dense_shape[d];
            } else {
                break;
            }
        }
        if inner_dims == 0 {
            return None;
        }
        // The outer dims must walk bcast's storage contiguously in
        // row-major order.
        let outer_ndims = ndims - inner_dims;
        let mut expected: isize = 1;
        for d in (0..outer_ndims).rev() {
            if bcast_strides[d] != expected {
                return None;
            }
            expected *= dense_shape[d] as isize;
        }
        let outer_count: usize = (0..outer_ndims).map(|d| dense_shape[d]).product();
        if outer_count == 0 || row_len == 0 {
            return None;
        }
        let bcast_scalar_base = bcast_layout.start_offset();
        // Bounds: kernel reads `bcast_storage[base..base+outer_count]`.
        if bcast_scalar_base.checked_add(outer_count)? > bcast_storage_elems {
            return None;
        }
        return Some(BroadcastView::PerRowScalar {
            outer_count,
            row_len,
            bcast_scalar_base,
        });
    }

    None
}

/// Execute a detected broadcast pattern for f32. Writes in-place into
/// the dense operand when unique; otherwise allocates a fresh
/// contiguous output.
///
/// `reversed` means the dense operand is the original *rhs* (swapped
/// orientation), so sub/div must compute `bcast OP dense` instead of
/// `dense OP bcast`.
#[cfg(feature = "simd")]
fn apply_broadcast_pattern_f32(
    mut dense: FlexTensor,
    bcast: &FlexTensor,
    simd_op: BinaryOp,
    pattern: BroadcastView,
    reversed: bool,
) -> FlexTensor {
    let numel = dense.layout().num_elements();
    let bcast_storage = bcast.storage::<f32>();

    if dense.is_unique() {
        let dst = &mut dense.storage_mut::<f32>()[..numel];
        run_broadcast_pattern_f32(dst, bcast_storage, simd_op, pattern, reversed);
        dense
    } else {
        // Copy the dense operand once, then apply the broadcast in
        // place. The memcpy is cheaper than the StridedIter fallback
        // it replaces.
        let mut out: Vec<f32> = dense.storage::<f32>()[..numel].to_vec();
        run_broadcast_pattern_f32(&mut out, bcast_storage, simd_op, pattern, reversed);
        make_tensor(out, dense.layout().shape().clone(), dense.dtype())
    }
}

/// Shared kernel: run the chosen broadcast pattern against a mutable
/// destination buffer (which already holds the dense operand's values)
/// and the broadcast operand's storage slice.
#[cfg(feature = "simd")]
fn run_broadcast_pattern_f32(
    dst: &mut [f32],
    bcast_storage: &[f32],
    simd_op: BinaryOp,
    pattern: BroadcastView,
    reversed: bool,
) {
    match pattern {
        BroadcastView::SharedRow {
            outer_count,
            row_len,
            bcast_row_offset,
        } => {
            let bcast_row: &[f32] = &bcast_storage[bcast_row_offset..bcast_row_offset + row_len];
            let total = outer_count * row_len;
            // One SIMD dispatch covers the whole outer walk. The kernel
            // keeps `bcast_row` in registers across rows for small row
            // lengths, and pays the macerator feature-detection cost
            // exactly once.
            let dst_full = &mut dst[..total];
            match (simd_op, reversed) {
                (BinaryOp::Add, _) => simd::add_shared_row_inplace_f32(dst_full, bcast_row),
                (BinaryOp::Sub, false) => simd::sub_shared_row_inplace_f32(dst_full, bcast_row),
                (BinaryOp::Sub, true) => simd::rsub_shared_row_inplace_f32(dst_full, bcast_row),
                (BinaryOp::Mul, _) => simd::mul_shared_row_inplace_f32(dst_full, bcast_row),
                (BinaryOp::Div, false) => simd::div_shared_row_inplace_f32(dst_full, bcast_row),
                (BinaryOp::Div, true) => simd::rdiv_shared_row_inplace_f32(dst_full, bcast_row),
            }
        }
        BroadcastView::PerRowScalar {
            outer_count,
            row_len,
            bcast_scalar_base,
        } => {
            let scalars = &bcast_storage[bcast_scalar_base..bcast_scalar_base + outer_count];
            // One monomorphized helper per op. The closure is statically
            // known at each call site so LLVM still autovectorizes the
            // inner scalar loop, and the outer op dispatch happens once.
            match (simd_op, reversed) {
                (BinaryOp::Add, _) => per_row_scalar_apply(dst, scalars, row_len, |a, b| a + b),
                (BinaryOp::Sub, false) => per_row_scalar_apply(dst, scalars, row_len, |a, b| a - b),
                (BinaryOp::Sub, true) => per_row_scalar_apply(dst, scalars, row_len, |a, b| b - a),
                (BinaryOp::Mul, _) => per_row_scalar_apply(dst, scalars, row_len, |a, b| a * b),
                (BinaryOp::Div, false) => per_row_scalar_apply(dst, scalars, row_len, |a, b| a / b),
                (BinaryOp::Div, true) => per_row_scalar_apply(dst, scalars, row_len, |a, b| b / a),
            }
        }
    }
}

/// Apply `dst[r * row_len + j] = op(dst[r * row_len + j], scalars[r])`
/// for `r in 0..scalars.len(), j in 0..row_len`. Generic over `Op` so
/// each call site gets a monomorphized, autovectorizable inner loop.
#[cfg(feature = "simd")]
#[inline]
fn per_row_scalar_apply<Op>(dst: &mut [f32], scalars: &[f32], row_len: usize, op: Op)
where
    Op: Fn(f32, f32) -> f32,
{
    for (i, &scalar) in scalars.iter().enumerate() {
        let start = i * row_len;
        for x in dst[start..start + row_len].iter_mut() {
            *x = op(*x, scalar);
        }
    }
}

/// Fallback when SIMD is disabled.
#[cfg(not(feature = "simd"))]
fn binary_op_f32<Op>(
    lhs: FlexTensor,
    rhs: FlexTensor,
    op: Op,
    _simd_hint: Option<BinaryOp>,
) -> FlexTensor
where
    Op: Fn(f32, f32) -> f32,
{
    binary_op_typed(lhs, &rhs, op)
}

/// Binary operation with in-place optimization for Pod types.
pub(crate) fn binary_op_typed<E, Op>(mut lhs: FlexTensor, rhs: &FlexTensor, op: Op) -> FlexTensor
where
    E: Element + bytemuck::Pod,
    Op: Fn(E, E) -> E,
{
    let rhs_storage: &[E] = rhs.storage();

    // In-place fast path: lhs unique, contiguous at offset 0, rhs contiguous
    if lhs.is_unique()
        && let (Some((0, l_end)), Some((r_start, r_end))) = (
            lhs.layout().contiguous_offsets(),
            rhs.layout().contiguous_offsets(),
        )
    {
        let lhs_storage: &mut [E] = lhs.storage_mut();
        let r_slice = &rhs_storage[r_start..r_end];
        for (l, &r) in lhs_storage[..l_end].iter_mut().zip(r_slice) {
            *l = op(*l, r);
        }
        return lhs;
    }

    // Allocating path
    let shape = lhs.layout().shape().clone();
    let dtype = lhs.dtype();
    let lhs_storage: &[E] = lhs.storage();

    let result: Vec<E> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        // Both contiguous (but lhs not at offset 0)
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            l_slice
                .iter()
                .zip(r_slice)
                .map(|(&a, &b)| op(a, b))
                .collect()
        }
        // Strided/broadcast fallback: collapse both layouts into a
        // joint loop nest with a specialized (autovectorizable) inner
        // loop. Only negative-stride (flipped) or over-rank layouts
        // fall through to the per-element StridedIter odometer.
        _ => {
            if let Some(result) =
                crate::zip::zip_map(lhs_storage, lhs.layout(), rhs_storage, rhs.layout(), &op)
            {
                result
            } else if lhs.layout().num_dims() == 2 {
                // 2D non-contiguous with negative strides (e.g. flipped)
                apply_2d_strided(lhs_storage, rhs_storage, lhs.layout(), rhs.layout(), &op)
            } else {
                let lhs_iter = StridedIter::new(lhs.layout());
                let rhs_iter = StridedIter::new(rhs.layout());
                lhs_iter
                    .zip(rhs_iter)
                    .map(|(li, ri)| op(lhs_storage[li], rhs_storage[ri]))
                    .collect()
            }
        }
    };

    make_tensor(result, shape, dtype)
}

/// Fast 2D strided binary operation using row-based iteration.
#[inline]
pub(crate) fn apply_2d_strided<E, R, Op>(
    lhs: &[E],
    rhs: &[E],
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    op: Op,
) -> Vec<R>
where
    E: Copy,
    Op: Fn(E, E) -> R,
{
    let (rows, cols, l_row_stride, l_col_stride) = lhs_layout.as_2d_strides().unwrap();
    let (_, _, r_row_stride, r_col_stride) = rhs_layout.as_2d_strides().unwrap();
    let l_offset = lhs_layout.start_offset() as isize;
    let r_offset = rhs_layout.start_offset() as isize;

    let mut result = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        let l_row_start = l_offset + row as isize * l_row_stride;
        let r_row_start = r_offset + row as isize * r_row_stride;
        for col in 0..cols {
            let l_idx = (l_row_start + col as isize * l_col_stride) as usize;
            let r_idx = (r_row_start + col as isize * r_col_stride) as usize;
            result.push(op(lhs[l_idx], rhs[r_idx]));
        }
    }

    result
}

/// Apply a scalar operation to each element of a tensor.
///
/// Attempts in-place mutation when tensor is contiguous at offset 0.
pub fn scalar_op<F32Op, F64Op>(
    tensor: FlexTensor,
    scalar: f64,
    f32_op: F32Op,
    f64_op: F64Op,
) -> FlexTensor
where
    F32Op: Fn(f32, f32) -> f32 + Copy,
    F64Op: Fn(f64, f64) -> f64 + Copy,
{
    let dtype = tensor.dtype();

    match dtype {
        DType::F32 => scalar_op_typed(tensor, scalar as f32, f32_op),
        DType::F64 => scalar_op_typed(tensor, scalar, f64_op),
        DType::F16 => {
            let scalar_f16 = f16::from_f32(scalar as f32);
            let s = scalar_f16.to_f32();
            scalar_op_typed(tensor, scalar_f16, |a: f16, _| {
                f16::from_f32(f32_op(a.to_f32(), s))
            })
        }
        DType::BF16 => {
            let scalar_bf16 = bf16::from_f32(scalar as f32);
            let s = scalar_bf16.to_f32();
            scalar_op_typed(tensor, scalar_bf16, |a: bf16, _| {
                bf16::from_f32(f32_op(a.to_f32(), s))
            })
        }
        _ => panic!("scalar_op: unsupported dtype {:?}", dtype),
    }
}

pub(crate) fn scalar_op_typed<E, Op>(mut tensor: FlexTensor, scalar: E, op: Op) -> FlexTensor
where
    E: Element + bytemuck::Pod,
    Op: Fn(E, E) -> E,
{
    // In-place fast path: unique, contiguous at offset 0
    if tensor.is_unique()
        && let Some((0, end)) = tensor.layout().contiguous_offsets()
    {
        let storage: &mut [E] = tensor.storage_mut();
        for x in storage[..end].iter_mut() {
            *x = op(*x, scalar);
        }
        return tensor;
    }

    // Allocating path
    let shape = tensor.layout().shape().clone();
    let dtype = tensor.dtype();
    let storage: &[E] = tensor.storage();

    let result: Vec<E> = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => storage[start..end].iter().map(|&x| op(x, scalar)).collect(),
        None => StridedIter::new(tensor.layout())
            .map(|i| op(storage[i], scalar))
            .collect(),
    };

    make_tensor(result, shape, dtype)
}

/// Helper to construct a tensor from result data.
fn make_tensor<E: bytemuck::Pod + Send + Sync>(
    data: Vec<E>,
    shape: Shape,
    dtype: DType,
) -> FlexTensor {
    let bytes = Bytes::from_elems(data);
    let layout = Layout::contiguous(shape);
    FlexTensor::new(bytes, layout, dtype)
}

/// Apply a binary operation element-wise to two integer tensors.
///
/// Supports all integer dtypes: I64, I32, I16, I8, U64, U32, U16, U8.
pub fn int_binary_op<Op>(lhs: FlexTensor, rhs: FlexTensor, op: Op) -> FlexTensor
where
    Op: Fn(i64, i64) -> i64 + Copy,
{
    debug_assert_eq!(lhs.dtype(), rhs.dtype(), "int_binary_op: dtype mismatch");

    // Broadcast tensors to the same shape if needed
    let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

    let dtype = lhs.dtype();

    match dtype {
        DType::I64 => binary_op_typed(lhs, &rhs, op),
        DType::I32 => binary_op_typed(lhs, &rhs, |a: i32, b: i32| op(a as i64, b as i64) as i32),
        DType::I16 => binary_op_typed(lhs, &rhs, |a: i16, b: i16| op(a as i64, b as i64) as i16),
        DType::I8 => binary_op_typed(lhs, &rhs, |a: i8, b: i8| op(a as i64, b as i64) as i8),
        // u64 values > i64::MAX wrap to negative i64. This is correct for
        // add/sub/mul/bitwise (two's complement). Div/rem are handled at the call site.
        DType::U64 => binary_op_typed(lhs, &rhs, |a: u64, b: u64| op(a as i64, b as i64) as u64),
        DType::U32 => binary_op_typed(lhs, &rhs, |a: u32, b: u32| op(a as i64, b as i64) as u32),
        DType::U16 => binary_op_typed(lhs, &rhs, |a: u16, b: u16| op(a as i64, b as i64) as u16),
        DType::U8 => binary_op_typed(lhs, &rhs, |a: u8, b: u8| op(a as i64, b as i64) as u8),
        _ => panic!("int_binary_op: unsupported dtype {:?}", dtype),
    }
}

/// Apply a scalar operation to each element of an integer tensor.
/// Note: scalar is truncated to target dtype (matches PyTorch).
pub fn int_scalar_op<Op>(tensor: FlexTensor, scalar: i64, op: Op) -> FlexTensor
where
    Op: Fn(i64, i64) -> i64 + Copy,
{
    let dtype = tensor.dtype();

    match dtype {
        DType::I64 => scalar_op_typed(tensor, scalar, op),
        DType::I32 => scalar_op_typed(tensor, scalar as i32, |a: i32, b: i32| {
            op(a as i64, b as i64) as i32
        }),
        DType::I16 => scalar_op_typed(tensor, scalar as i16, |a: i16, b: i16| {
            op(a as i64, b as i64) as i16
        }),
        DType::I8 => scalar_op_typed(tensor, scalar as i8, |a: i8, b: i8| {
            op(a as i64, b as i64) as i8
        }),
        DType::U64 => scalar_op_typed(tensor, scalar as u64, |a: u64, b: u64| {
            op(a as i64, b as i64) as u64
        }),
        DType::U32 => scalar_op_typed(tensor, scalar as u32, |a: u32, b: u32| {
            op(a as i64, b as i64) as u32
        }),
        DType::U16 => scalar_op_typed(tensor, scalar as u16, |a: u16, b: u16| {
            op(a as i64, b as i64) as u16
        }),
        DType::U8 => scalar_op_typed(tensor, scalar as u8, |a: u8, b: u8| {
            op(a as i64, b as i64) as u8
        }),
        _ => panic!("int_scalar_op: unsupported dtype {:?}", dtype),
    }
}

// Tests kept here exercise flex-specific behavior of `binary_op` /
// `scalar_op`: non-contiguous (transposed/narrowed/permuted) strides,
// flex f16/bf16 half-precision storage paths, and broadcast patterns
// that probe the flex layout system. Plain contiguous add/sub/mul/div
// and scalar-op smoke tests have been dropped in favor of the
// equivalent coverage in burn-backend-tests, which exercises every
// backend. When adding new tests, keep them here only if they probe
// flex-internal dispatch; otherwise add them to
// crates/burn-backend-tests/tests/tensor/float/ops/.
#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use alloc::vec;
    use burn_backend::{TensorData, Tolerance};

    // ===================
    // F16 tests
    // ===================

    #[test]
    fn test_binary_add_f16() {
        let a_vals: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b_vals: Vec<f16> = vec![5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(b_vals, vec![2, 2]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let expected: Vec<f16> = vec![6.0, 8.0, 10.0, 12.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(f16::from_f32(0.01)),
        );
    }

    #[test]
    fn test_binary_mul_f16() {
        let a_vals: Vec<f16> = vec![2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b_vals: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(b_vals, vec![2, 2]));

        let result = binary_op(a, b, |x, y| x * y, |x, y| x * y, None);
        let expected: Vec<f16> = vec![2.0, 6.0, 12.0, 20.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(f16::from_f32(0.01)),
        );
    }

    #[test]
    fn test_binary_f16_transposed() {
        let a_vals: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let b_vals: Vec<f16> = vec![10.0, 20.0, 30.0, 40.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();

        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![2, 2])).transpose(0, 1);
        let b = FlexTensor::from_data(TensorData::new(b_vals, vec![2, 2])).transpose(0, 1);

        // a_t = [[1,3], [2,4]], b_t = [[10,30], [20,40]]
        // result = [[11,33], [22,44]]
        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let expected: Vec<f16> = vec![11.0, 33.0, 22.0, 44.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(f16::from_f32(0.1)),
        );
    }

    #[test]
    fn test_scalar_f16() {
        let a_vals: Vec<f16> = vec![1.0, 2.0, 3.0].into_iter().map(f16::from_f32).collect();
        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![3]));

        let result = scalar_op(a, 10.0, |x, y| x + y, |x, y| x + y);
        let expected: Vec<f16> = vec![11.0, 12.0, 13.0]
            .into_iter()
            .map(f16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![3]),
            Tolerance::absolute(f16::from_f32(0.01)),
        );
    }

    // ===================
    // BF16 tests
    // ===================

    #[test]
    fn test_binary_add_bf16() {
        let a_vals: Vec<bf16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let b_vals: Vec<bf16> = vec![5.0, 6.0, 7.0, 8.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();

        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(b_vals, vec![2, 2]));

        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let expected: Vec<bf16> = vec![6.0, 8.0, 10.0, 12.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(bf16::from_f32(0.1)),
        );
    }

    #[test]
    fn test_binary_mul_bf16() {
        let a_vals: Vec<bf16> = vec![2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let b_vals: Vec<bf16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();

        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![2, 2]));
        let b = FlexTensor::from_data(TensorData::new(b_vals, vec![2, 2]));

        let result = binary_op(a, b, |x, y| x * y, |x, y| x * y, None);
        let expected: Vec<bf16> = vec![2.0, 6.0, 12.0, 20.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(bf16::from_f32(0.1)),
        );
    }

    #[test]
    fn test_binary_bf16_transposed() {
        let a_vals: Vec<bf16> = vec![1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let b_vals: Vec<bf16> = vec![10.0, 20.0, 30.0, 40.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();

        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![2, 2])).transpose(0, 1);
        let b = FlexTensor::from_data(TensorData::new(b_vals, vec![2, 2])).transpose(0, 1);

        // a_t = [[1,3], [2,4]], b_t = [[10,30], [20,40]]
        // result = [[11,33], [22,44]]
        let result = binary_op(a, b, |x, y| x + y, |x, y| x + y, None);
        let expected: Vec<bf16> = vec![11.0, 33.0, 22.0, 44.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![2, 2]),
            Tolerance::absolute(bf16::from_f32(0.5)),
        );
    }

    #[test]
    fn test_scalar_bf16() {
        let a_vals: Vec<bf16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![3]));

        let result = scalar_op(a, 10.0, |x, y| x + y, |x, y| x + y);
        let expected: Vec<bf16> = vec![11.0, 12.0, 13.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        result.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![3]),
            Tolerance::absolute(bf16::from_f32(0.1)),
        );
    }

    #[test]
    fn test_scalar_f16_non_representable() {
        // 0.1 is not exactly representable in f16; verify the scalar is rounded
        // to f16 precision before the op (matching dtype semantics).
        let a_vals: Vec<f16> = vec![1.0, 2.0, 3.0].into_iter().map(f16::from_f32).collect();
        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![3]));

        let s_f16 = f16::from_f32(0.1);
        let result = scalar_op(a, 0.1, |x, y| x * y, |x, y| x * y);
        let expected: Vec<f16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(|v| f16::from_f32(f16::from_f32(v).to_f32() * s_f16.to_f32()))
            .collect();
        result.into_data().assert_approx_eq::<f16>(
            &TensorData::new(expected, vec![3]),
            Tolerance::absolute(f16::from_f32(0.001)),
        );
    }

    #[test]
    fn test_scalar_bf16_non_representable() {
        // 1.1 is not exactly representable in bf16; verify dtype rounding.
        let a_vals: Vec<bf16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let a = FlexTensor::from_data(TensorData::new(a_vals, vec![3]));

        let s_bf16 = bf16::from_f32(1.1);
        let result = scalar_op(a, 1.1, |x, y| x * y, |x, y| x * y);
        let expected: Vec<bf16> = vec![1.0, 2.0, 3.0]
            .into_iter()
            .map(|v| bf16::from_f32(bf16::from_f32(v).to_f32() * s_bf16.to_f32()))
            .collect();
        result.into_data().assert_approx_eq::<bf16>(
            &TensorData::new(expected, vec![3]),
            Tolerance::absolute(bf16::from_f32(0.01)),
        );
    }

    // ============================================================================
    // Broadcast binary-op fast paths
    // ============================================================================

    /// Shared-row broadcast: 1-D gamma reshaped + expanded, with the
    /// size-1 outer dim exemption in play.
    #[test]
    fn test_binary_shared_row_broadcast_f32() {
        let a = FlexTensor::from_data(TensorData::new(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![1, 3, 4],
        ));
        // 1D gamma broadcast over rows.
        let gamma =
            FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![4]));
        let gamma_unsqueezed = gamma.reshape(Shape::from(vec![1, 1, 4]));
        let result = binary_op(
            a,
            gamma_unsqueezed,
            |a, b| a * b,
            |a, b| a * b,
            Some(BinaryOp::Mul),
        );
        let data = result.into_data();
        let expected = vec![
            10.0f32, 40.0, 90.0, 160.0, 50.0, 120.0, 210.0, 320.0, 90.0, 200.0, 330.0, 480.0,
        ];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    /// Per-row scalar broadcast: `[1, 3, 4] - mean_dim(-1)` shape,
    /// rhs expands to strides `[3, 1, 0]`.
    #[test]
    fn test_binary_per_row_scalar_broadcast_f32() {
        let a = FlexTensor::from_data(TensorData::new(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
            ],
            vec![1, 3, 4],
        ));
        // Scalars shaped like `mean_dim(-1)` output.
        let mean = FlexTensor::from_data(TensorData::new(vec![2.5f32, 25.0, 250.0], vec![1, 3, 1]));
        let result = binary_op(a, mean, |a, b| a - b, |a, b| a - b, Some(BinaryOp::Sub));
        let data = result.into_data();
        let expected = vec![
            -1.5f32, -0.5, 0.5, 1.5, -15.0, -5.0, 5.0, 15.0, -150.0, -50.0, 50.0, 150.0,
        ];
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    /// Non-contig lhs + shared-row broadcast: must materialize lhs
    /// contiguous then dispatch to the shared-row kernel.
    #[test]
    fn test_binary_permuted_lhs_broadcast_rhs() {
        let a = FlexTensor::from_data(TensorData::new(
            (0..24).map(|i| i as f32).collect::<Vec<_>>(),
            vec![2, 3, 4],
        ));
        let a_permuted = a.transpose(1, 2); // shape [2, 4, 3]
        let gamma = FlexTensor::from_data(TensorData::new(vec![1.0f32, 10.0, 100.0], vec![3]));
        let gamma_expanded = gamma.reshape(Shape::from(vec![1, 1, 3]));

        let result = binary_op(
            a_permuted,
            gamma_expanded,
            |a, b| a * b,
            |a, b| a * b,
            Some(BinaryOp::Mul),
        );

        // Compare against a naive reference computed on the permuted
        // values.
        let reference: Vec<f32> = {
            // original[b, r, c] = b*12 + r*4 + c
            // permuted[b, c, r] = original[b, r, c]
            // result[b, c, r] = permuted[b, c, r] * gamma_row[r]
            let gamma_vals = [1.0f32, 10.0, 100.0];
            let mut out = Vec::with_capacity(24);
            for b in 0..2 {
                for c in 0..4 {
                    for r in 0..3 {
                        let orig_val = (b * 12 + r * 4 + c) as f32;
                        out.push(orig_val * gamma_vals[r]);
                    }
                }
            }
            out
        };

        let data = result.into_data();
        assert_eq!(data.as_slice::<f32>().unwrap(), reference.as_slice());
    }

    /// Non-contig lhs + per-row-scalar broadcast-sub.
    #[test]
    fn test_binary_permuted_lhs_per_row_scalar_sub() {
        let a = FlexTensor::from_data(TensorData::new(
            (1..=24).map(|i| i as f32).collect::<Vec<_>>(),
            vec![2, 3, 4],
        ));
        let a_permuted = a.transpose(1, 2); // shape [2, 4, 3]

        // Per-row scalar in the permuted layout: shape [2, 4, 1].
        let mean = FlexTensor::from_data(TensorData::new(
            (0..8).map(|i| i as f32).collect::<Vec<_>>(),
            vec![2, 4, 1],
        ));

        let result = binary_op(
            a_permuted,
            mean,
            |a, b| a - b,
            |a, b| a - b,
            Some(BinaryOp::Sub),
        );

        // Reference computation.
        let reference: Vec<f32> = {
            let mut out = Vec::with_capacity(24);
            for b in 0..2 {
                for c in 0..4 {
                    let mean_val = (b * 4 + c) as f32;
                    for r in 0..3 {
                        let orig_val = (b * 12 + r * 4 + c + 1) as f32;
                        out.push(orig_val - mean_val);
                    }
                }
            }
            out
        };

        let data = result.into_data();
        assert_eq!(data.as_slice::<f32>().unwrap(), reference.as_slice());
    }

    /// Exercise every `(op, pattern)` combination of the broadcast fast path:
    /// Add/Sub/Mul/Div crossed with SharedRow and PerRowScalar. The existing
    /// targeted tests only cover a subset, so a sign error in
    /// `div_shared_row_inplace_f32` or `add_per_row_scalar` would ship green.
    #[test]
    fn test_binary_broadcast_all_ops_and_patterns_f32() {
        fn build_shared() -> (FlexTensor, FlexTensor) {
            let a = FlexTensor::from_data(TensorData::new(
                vec![4.0f32, 8.0, 12.0, 20.0, 30.0, 60.0],
                vec![2, 3],
            ));
            let b = FlexTensor::from_data(TensorData::new(vec![2.0f32, 4.0, 3.0], vec![3]))
                .reshape(Shape::from(vec![1, 3]));
            (a, b)
        }
        fn build_perrow() -> (FlexTensor, FlexTensor) {
            let a = FlexTensor::from_data(TensorData::new(
                vec![4.0f32, 8.0, 12.0, 20.0, 30.0, 60.0],
                vec![2, 3],
            ));
            let b = FlexTensor::from_data(TensorData::new(vec![2.0f32, 5.0], vec![2, 1]));
            (a, b)
        }

        let run = |name: &str,
                   build: fn() -> (FlexTensor, FlexTensor),
                   simd_op: BinaryOp,
                   op_fn: fn(f32, f32) -> f32,
                   expected: &[f32]| {
            let (a, b) = build();
            let result = binary_op(
                a,
                b,
                op_fn,
                |x: f64, y: f64| op_fn(x as f32, y as f32) as f64,
                Some(simd_op),
            );
            let data = result.into_data();
            assert_eq!(
                data.as_slice::<f32>().unwrap(),
                expected,
                "case {name} produced wrong values"
            );
        };

        // SharedRow expected: lhs[i][j] OP rhs[j]
        run(
            "shared_add",
            build_shared,
            BinaryOp::Add,
            |a, b| a + b,
            &[6.0, 12.0, 15.0, 22.0, 34.0, 63.0],
        );
        run(
            "shared_sub",
            build_shared,
            BinaryOp::Sub,
            |a, b| a - b,
            &[2.0, 4.0, 9.0, 18.0, 26.0, 57.0],
        );
        run(
            "shared_mul",
            build_shared,
            BinaryOp::Mul,
            |a, b| a * b,
            &[8.0, 32.0, 36.0, 40.0, 120.0, 180.0],
        );
        run(
            "shared_div",
            build_shared,
            BinaryOp::Div,
            |a, b| a / b,
            &[2.0, 2.0, 4.0, 10.0, 7.5, 20.0],
        );
        // PerRowScalar expected: lhs[i][j] OP rhs[i]
        run(
            "perrow_add",
            build_perrow,
            BinaryOp::Add,
            |a, b| a + b,
            &[6.0, 10.0, 14.0, 25.0, 35.0, 65.0],
        );
        run(
            "perrow_sub",
            build_perrow,
            BinaryOp::Sub,
            |a, b| a - b,
            &[2.0, 6.0, 10.0, 15.0, 25.0, 55.0],
        );
        run(
            "perrow_mul",
            build_perrow,
            BinaryOp::Mul,
            |a, b| a * b,
            &[8.0, 16.0, 24.0, 100.0, 150.0, 300.0],
        );
        run(
            "perrow_div",
            build_perrow,
            BinaryOp::Div,
            |a, b| a / b,
            &[2.0, 4.0, 6.0, 4.0, 6.0, 12.0],
        );
    }

    /// Non-unique lhs: `apply_broadcast_pattern_f32` takes the allocating
    /// branch instead of writing in place. Clone the lhs so its Arc refcount
    /// is > 1, then run a broadcast op and verify the result matches the
    /// unique path. Without this test, a regression in the allocating branch
    /// would only fire on shared-lhs call sites which are rare in bench code.
    #[test]
    fn test_binary_broadcast_non_unique_lhs_f32() {
        let a = FlexTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        ));
        let _keep_alive = a.clone(); // bump Arc refcount so lhs is shared
        let b = FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0], vec![3]))
            .reshape(Shape::from(vec![1, 3]));
        let result = binary_op(a, b, |a, b| a + b, |a, b| a + b, Some(BinaryOp::Add));
        let data = result.into_data();
        assert_eq!(
            data.as_slice::<f32>().unwrap(),
            &[11.0f32, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }

    /// Fully-broadcast scalar: rhs strides all 0, PerRowScalar with
    /// empty outer walk, applies one scalar across the whole dst.
    #[test]
    fn test_binary_fully_broadcast_scalar_f32() {
        let a = FlexTensor::from_data(TensorData::new(
            (0..12).map(|i| i as f32).collect::<Vec<_>>(),
            vec![2, 2, 3],
        ));
        // 1-element tensor expanded to lhs's full shape. All strides
        // become 0.
        let scalar_tensor = FlexTensor::from_data(TensorData::new(vec![100.0f32], [1]));
        let scalar_expanded = crate::ops::expand::expand(scalar_tensor, Shape::from(vec![2, 2, 3]));
        // Sanity check: every stride is 0.
        assert!(scalar_expanded.layout().strides().iter().all(|&s| s == 0));

        let result = binary_op(
            a,
            scalar_expanded,
            |a, b| a + b,
            |a, b| a + b,
            Some(BinaryOp::Add),
        );

        let expected: Vec<f32> = (0..12).map(|i| i as f32 + 100.0).collect();
        let data = result.into_data();
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    // ============================================================================
    // Swapped-operand fast paths (broadcast operand on the LEFT)
    // ============================================================================

    /// Every `(op, pattern)` combination with the *broadcast operand on
    /// the left*: `[1,3] OP [2,3]` (SharedRow) and `[2,1] OP [2,3]`
    /// (PerRowScalar). Sub and Div exercise the reversed kernels
    /// (`rsub`/`rdiv`); a kernel that silently computed `rhs OP lhs`
    /// would pass for Add/Mul but fail here.
    #[test]
    fn test_binary_broadcast_lhs_all_ops_and_patterns_f32() {
        // SharedRow: lhs row [2,4,8] broadcast over 2 rows of rhs.
        fn build_shared() -> (FlexTensor, FlexTensor) {
            let a = FlexTensor::from_data(TensorData::new(vec![2.0f32, 4.0, 8.0], vec![3]))
                .reshape(Shape::from(vec![1, 3]));
            let b = FlexTensor::from_data(TensorData::new(
                vec![1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0],
                vec![2, 3],
            ));
            (a, b)
        }
        // PerRowScalar: lhs one scalar per row, broadcast over columns.
        fn build_perrow() -> (FlexTensor, FlexTensor) {
            let a = FlexTensor::from_data(TensorData::new(vec![12.0f32, 100.0], vec![2, 1]));
            let b = FlexTensor::from_data(TensorData::new(
                vec![1.0f32, 2.0, 4.0, 10.0, 20.0, 50.0],
                vec![2, 3],
            ));
            (a, b)
        }

        let run = |name: &str,
                   build: fn() -> (FlexTensor, FlexTensor),
                   simd_op: BinaryOp,
                   op_fn: fn(f32, f32) -> f32,
                   expected: &[f32]| {
            let (a, b) = build();
            let result = binary_op(
                a,
                b,
                op_fn,
                |x: f64, y: f64| op_fn(x as f32, y as f32) as f64,
                Some(simd_op),
            );
            let data = result.into_data();
            assert_eq!(
                data.as_slice::<f32>().unwrap(),
                expected,
                "case {name} produced wrong values"
            );
        };

        // SharedRow expected: lhs[j] OP rhs[i][j]
        run(
            "shared_add",
            build_shared,
            BinaryOp::Add,
            |a, b| a + b,
            &[3.0, 6.0, 12.0, 10.0, 20.0, 40.0],
        );
        run(
            "shared_sub",
            build_shared,
            BinaryOp::Sub,
            |a, b| a - b,
            &[1.0, 2.0, 4.0, -6.0, -12.0, -24.0],
        );
        run(
            "shared_mul",
            build_shared,
            BinaryOp::Mul,
            |a, b| a * b,
            &[2.0, 8.0, 32.0, 16.0, 64.0, 256.0],
        );
        run(
            "shared_div",
            build_shared,
            BinaryOp::Div,
            |a, b| a / b,
            &[2.0, 2.0, 2.0, 0.25, 0.25, 0.25],
        );
        // PerRowScalar expected: lhs[i] OP rhs[i][j]
        run(
            "perrow_add",
            build_perrow,
            BinaryOp::Add,
            |a, b| a + b,
            &[13.0, 14.0, 16.0, 110.0, 120.0, 150.0],
        );
        run(
            "perrow_sub",
            build_perrow,
            BinaryOp::Sub,
            |a, b| a - b,
            &[11.0, 10.0, 8.0, 90.0, 80.0, 50.0],
        );
        run(
            "perrow_mul",
            build_perrow,
            BinaryOp::Mul,
            |a, b| a * b,
            &[12.0, 24.0, 48.0, 1000.0, 2000.0, 5000.0],
        );
        run(
            "perrow_div",
            build_perrow,
            BinaryOp::Div,
            |a, b| a / b,
            &[12.0, 6.0, 3.0, 10.0, 5.0, 2.0],
        );
    }

    /// 3-D swapped SharedRow: `gamma[1,1,4] - x[2,3,4]`, the mirror of
    /// the layer-norm `x * gamma.unsqueeze()` shape with a
    /// non-commutative op.
    #[test]
    fn test_binary_broadcast_lhs_3d_shared_row_sub_f32() {
        let gamma = FlexTensor::from_data(TensorData::new(
            vec![100.0f32, 200.0, 300.0, 400.0],
            vec![4],
        ))
        .reshape(Shape::from(vec![1, 1, 4]));
        let x = FlexTensor::from_data(TensorData::new(
            (0..24).map(|i| i as f32).collect::<Vec<_>>(),
            vec![2, 3, 4],
        ));

        let result = binary_op(gamma, x, |a, b| a - b, |a, b| a - b, Some(BinaryOp::Sub));

        let gamma_vals = [100.0f32, 200.0, 300.0, 400.0];
        let expected: Vec<f32> = (0..24).map(|i| gamma_vals[i % 4] - i as f32).collect();
        let data = result.into_data();
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    /// Swapped in-place path: lhs is contiguous but shared (refcount >
    /// 1), rhs is unique -> the op writes into rhs's storage, using the
    /// reversed kernels for sub/div.
    #[test]
    fn test_binary_swapped_inplace_shared_lhs_f32() {
        let run = |simd_op: BinaryOp, op_fn: fn(f32, f32) -> f32, expected: &[f32]| {
            let a =
                FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2]));
            let _keep_alive = a.clone(); // lhs shared: in-place-on-lhs path can't fire
            let b = FlexTensor::from_data(TensorData::new(vec![2.0f32, 4.0, 5.0, 8.0], vec![2, 2]));
            let result = binary_op(
                a,
                b,
                op_fn,
                |x: f64, y: f64| op_fn(x as f32, y as f32) as f64,
                Some(simd_op),
            );
            let data = result.into_data();
            assert_eq!(data.as_slice::<f32>().unwrap(), expected);
        };

        run(BinaryOp::Add, |a, b| a + b, &[12.0, 24.0, 35.0, 48.0]);
        run(BinaryOp::Sub, |a, b| a - b, &[8.0, 16.0, 25.0, 32.0]);
        run(BinaryOp::Mul, |a, b| a * b, &[20.0, 80.0, 150.0, 320.0]);
        run(BinaryOp::Div, |a, b| a / b, &[5.0, 5.0, 6.0, 5.0]);
    }

    /// Swapped broadcast with a *shared* dense rhs: the allocating
    /// branch of `apply_broadcast_pattern_f32` must also honor the
    /// reversed op.
    #[test]
    fn test_binary_broadcast_lhs_non_unique_rhs_sub_f32() {
        let a = FlexTensor::from_data(TensorData::new(vec![2.0f32, 4.0, 8.0], vec![3]))
            .reshape(Shape::from(vec![1, 3]));
        let b = FlexTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 4.0, 8.0, 16.0, 32.0],
            vec![2, 3],
        ));
        let _keep_alive = b.clone(); // rhs shared: forces the allocating branch
        let result = binary_op(a, b, |a, b| a - b, |a, b| a - b, Some(BinaryOp::Sub));
        let data = result.into_data();
        assert_eq!(
            data.as_slice::<f32>().unwrap(),
            &[1.0f32, 2.0, 4.0, -6.0, -12.0, -24.0]
        );
    }

    /// Fully-broadcast scalar on the LEFT: `100 / x` with the scalar
    /// expanded to x's shape (all strides 0). Lands in the swapped
    /// PerRowScalar path with a single outer scalar and a reversed div.
    #[test]
    fn test_binary_fully_broadcast_scalar_lhs_div_f32() {
        let scalar_tensor = FlexTensor::from_data(TensorData::new(vec![100.0f32], [1]));
        let scalar_expanded = crate::ops::expand::expand(scalar_tensor, Shape::from(vec![2, 2, 3]));
        assert!(scalar_expanded.layout().strides().iter().all(|&s| s == 0));

        let x = FlexTensor::from_data(TensorData::new(
            (1..=12).map(|i| i as f32).collect::<Vec<_>>(),
            vec![2, 2, 3],
        ));

        let result = binary_op(
            scalar_expanded,
            x,
            |a, b| a / b,
            |a, b| a / b,
            Some(BinaryOp::Div),
        );

        let expected: Vec<f32> = (1..=12).map(|i| 100.0 / i as f32).collect();
        let data = result.into_data();
        assert_eq!(data.as_slice::<f32>().unwrap(), expected.as_slice());
    }

    /// Broadcast lhs + permuted rhs (`mean - x.permute(...)`): the
    /// mirror materialization must copy rhs contiguous so the swapped
    /// PerRowScalar path can fire, with the reversed sub kernel.
    #[test]
    fn test_binary_broadcast_lhs_permuted_rhs_sub_f32() {
        // mean[b, c] laid out as [2, 4, 1].
        let mean = FlexTensor::from_data(TensorData::new(
            (0..8).map(|i| (i * 10) as f32).collect::<Vec<_>>(),
            vec![2, 4, 1],
        ));
        let x = FlexTensor::from_data(TensorData::new(
            (1..=24).map(|i| i as f32).collect::<Vec<_>>(),
            vec![2, 3, 4],
        ));
        let x_permuted = x.transpose(1, 2); // shape [2, 4, 3], non-contiguous

        let result = binary_op(
            mean,
            x_permuted,
            |a, b| a - b,
            |a, b| a - b,
            Some(BinaryOp::Sub),
        );

        // result[b, c, r] = mean[b, c] - x[b, r, c]
        let reference: Vec<f32> = {
            let mut out = Vec::with_capacity(24);
            for b in 0..2 {
                for c in 0..4 {
                    let mean_val = ((b * 4 + c) * 10) as f32;
                    for r in 0..3 {
                        let x_val = (b * 12 + r * 4 + c + 1) as f32;
                        out.push(mean_val - x_val);
                    }
                }
            }
            out
        };

        let data = result.into_data();
        assert_eq!(data.as_slice::<f32>().unwrap(), reference.as_slice());
    }
}
