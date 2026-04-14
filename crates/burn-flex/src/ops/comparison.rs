//! Comparison operations returning boolean tensors.

use alloc::boxed::Box;
#[cfg(feature = "simd")]
use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{BoolDType, BoolStore, Bytes, Shape, bf16, f16};
use bytemuck::Pod;

use crate::strided_index::StridedIter;
use crate::{FlexTensor, Layout};

use crate::simd;

/// Comparison operation type for SIMD dispatch.
pub use simd::CmpOp as CompareOp;

/// Compare two tensors element-wise, returning a boolean tensor with the
/// requested output dtype.
pub fn compare<F32Cmp, F64Cmp>(
    lhs: FlexTensor,
    rhs: FlexTensor,
    out_dtype: BoolDType,
    f32_cmp: F32Cmp,
    f64_cmp: F64Cmp,
    simd_hint: Option<CompareOp>,
) -> FlexTensor
where
    F32Cmp: Fn(f32, f32) -> bool + Copy,
    F64Cmp: Fn(f64, f64) -> bool + Copy,
{
    debug_assert_eq!(lhs.dtype(), rhs.dtype(), "compare: dtype mismatch");

    // Broadcast to same shape if needed
    let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

    let dtype = lhs.dtype();

    match dtype {
        DType::F32 => compare_f32(lhs, &rhs, out_dtype, f32_cmp, simd_hint),
        DType::F64 => compare_typed(lhs, &rhs, out_dtype, f64_cmp),
        DType::F16 => compare_typed(lhs, &rhs, out_dtype, |a: f16, b: f16| {
            f32_cmp(a.to_f32(), b.to_f32())
        }),
        DType::BF16 => compare_typed(lhs, &rhs, out_dtype, |a: bf16, b: bf16| {
            f32_cmp(a.to_f32(), b.to_f32())
        }),
        _ => panic!("compare: unsupported dtype {:?}", dtype),
    }
}

/// Specialized comparison for f32 with SIMD fast path.
#[cfg(feature = "simd")]
fn compare_f32<Cmp>(
    lhs: FlexTensor,
    rhs: &FlexTensor,
    out_dtype: BoolDType,
    cmp: Cmp,
    simd_hint: Option<CompareOp>,
) -> FlexTensor
where
    Cmp: Fn(f32, f32) -> bool,
{
    // SIMD fast path: both tensors contiguous
    if let (Some((l_start, l_end)), Some((r_start, r_end))) = (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) && let Some(simd_op) = simd_hint
    {
        let shape = lhs.layout().shape().clone();
        let lhs_storage: &[f32] = lhs.storage();
        let rhs_storage: &[f32] = rhs.storage();

        let l_slice = &lhs_storage[l_start..l_end];
        let r_slice = &rhs_storage[r_start..r_end];

        let mut result = vec![0u8; l_slice.len()];
        simd::cmp_f32(l_slice, r_slice, &mut result, simd_op);

        return make_bool_tensor(result, shape, out_dtype);
    }

    // Optimized broadcast path for outer-product style broadcasting
    // Pattern: [N, 1] vs [1, M] -> [N, M] where one has stride 0 in inner dim
    if lhs.layout().num_dims() == 2
        && let Some(simd_op) = simd_hint
        && let Some((result, shape)) = try_broadcast_cmp_f32(&lhs, rhs, simd_op)
    {
        return make_bool_tensor(result, shape, out_dtype);
    }

    // Fallback to generic path
    compare_typed(lhs, rhs, out_dtype, cmp)
}

/// Try optimized outer-product style broadcast comparison.
/// Returns Some((result, shape)) if the pattern matches.
#[cfg(feature = "simd")]
fn try_broadcast_cmp_f32(
    lhs: &FlexTensor,
    rhs: &FlexTensor,
    op: simd::CmpOp,
) -> Option<(Vec<u8>, Shape)> {
    let lhs_strides = lhs.layout().strides();
    let rhs_strides = rhs.layout().strides();
    let shape = lhs.layout().shape().clone();
    let [rows, cols] = shape[..] else {
        return None;
    };

    // Pattern 1: lhs has stride 0 in dim 1 (column broadcast), rhs contiguous
    // lhs[i,j] = lhs_data[i*stride], rhs[i,j] = rhs_data[i*cols + j]
    if lhs_strides[1] == 0 && rhs_strides == [cols as isize, 1] {
        let lhs_storage: &[f32] = lhs.storage();
        let rhs_storage: &[f32] = rhs.storage();
        let l_offset = lhs.layout().start_offset() as isize;
        let l_stride = lhs_strides[0];
        let r_offset = rhs.layout().start_offset();

        let mut result = vec![0u8; rows * cols];
        for row in 0..rows {
            let a_val = lhs_storage[(l_offset + row as isize * l_stride) as usize];
            let r_row_start = r_offset + row * cols;
            let r_slice = &rhs_storage[r_row_start..r_row_start + cols];
            let out_start = row * cols;
            simd::cmp_scalar_f32(
                r_slice,
                a_val,
                &mut result[out_start..out_start + cols],
                swap_cmp_op(op),
            );
        }
        return Some((result, shape));
    }

    // Pattern 2: rhs has stride 0 in dim 0 (row broadcast), lhs contiguous
    // lhs[i,j] = lhs_data[i*cols + j], rhs[i,j] = rhs_data[j*stride]
    if rhs_strides[0] == 0 && lhs_strides == [cols as isize, 1] {
        let lhs_storage: &[f32] = lhs.storage();
        let rhs_storage: &[f32] = rhs.storage();
        let l_offset = lhs.layout().start_offset();
        let r_offset = rhs.layout().start_offset() as isize;
        let r_stride = rhs_strides[1];

        // Build the broadcast rhs values once
        let rhs_row: Vec<f32> = (0..cols)
            .map(|j| rhs_storage[(r_offset + j as isize * r_stride) as usize])
            .collect();

        let mut result = vec![0u8; rows * cols];
        for row in 0..rows {
            let l_row_start = l_offset + row * cols;
            let l_slice = &lhs_storage[l_row_start..l_row_start + cols];
            let out_start = row * cols;
            // Compare row with broadcast values
            for (j, (&lv, &rv)) in l_slice.iter().zip(rhs_row.iter()).enumerate() {
                result[out_start + j] = match op {
                    simd::CmpOp::Gt => (lv > rv) as u8,
                    simd::CmpOp::Ge => (lv >= rv) as u8,
                    simd::CmpOp::Lt => (lv < rv) as u8,
                    simd::CmpOp::Le => (lv <= rv) as u8,
                    simd::CmpOp::Eq => (lv == rv) as u8,
                    simd::CmpOp::Ne => (lv != rv) as u8,
                };
            }
        }
        return Some((result, shape));
    }

    // Pattern 3: Outer product - lhs stride 0 in dim 1, rhs stride 0 in dim 0
    // This is the [N,1] vs [1,M] case
    if lhs_strides[1] == 0 && rhs_strides[0] == 0 {
        let lhs_storage: &[f32] = lhs.storage();
        let rhs_storage: &[f32] = rhs.storage();
        let l_offset = lhs.layout().start_offset() as isize;
        let l_stride = lhs_strides[0];
        let r_offset = rhs.layout().start_offset() as isize;
        let r_stride = rhs_strides[1];

        // Build the broadcast rhs row once
        let rhs_row: Vec<f32> = (0..cols)
            .map(|j| rhs_storage[(r_offset + j as isize * r_stride) as usize])
            .collect();

        let mut result = vec![0u8; rows * cols];
        for row in 0..rows {
            let a_val = lhs_storage[(l_offset + row as isize * l_stride) as usize];
            let out_start = row * cols;
            simd::cmp_scalar_f32(
                &rhs_row,
                a_val,
                &mut result[out_start..out_start + cols],
                swap_cmp_op(op),
            );
        }
        return Some((result, shape));
    }

    None
}

/// Swap comparison operation for reversed operand order.
#[cfg(feature = "simd")]
fn swap_cmp_op(op: simd::CmpOp) -> simd::CmpOp {
    match op {
        simd::CmpOp::Gt => simd::CmpOp::Lt, // a > b becomes b < a
        simd::CmpOp::Ge => simd::CmpOp::Le,
        simd::CmpOp::Lt => simd::CmpOp::Gt,
        simd::CmpOp::Le => simd::CmpOp::Ge,
        simd::CmpOp::Eq => simd::CmpOp::Eq, // symmetric
        simd::CmpOp::Ne => simd::CmpOp::Ne,
    }
}

/// Fallback when SIMD is disabled.
#[cfg(not(feature = "simd"))]
fn compare_f32<Cmp>(
    lhs: FlexTensor,
    rhs: &FlexTensor,
    out_dtype: BoolDType,
    cmp: Cmp,
    _simd_hint: Option<CompareOp>,
) -> FlexTensor
where
    Cmp: Fn(f32, f32) -> bool,
{
    compare_typed(lhs, rhs, out_dtype, cmp)
}

/// Compare tensor with scalar, returning a boolean tensor with the requested
/// output dtype.
pub fn compare_elem<F32Cmp, F64Cmp>(
    lhs: FlexTensor,
    rhs: f64,
    out_dtype: BoolDType,
    f32_cmp: F32Cmp,
    f64_cmp: F64Cmp,
    simd_hint: Option<CompareOp>,
) -> FlexTensor
where
    F32Cmp: Fn(f32, f32) -> bool + Copy,
    F64Cmp: Fn(f64, f64) -> bool + Copy,
{
    let dtype = lhs.dtype();

    match dtype {
        DType::F32 => compare_elem_f32(lhs, rhs as f32, out_dtype, f32_cmp, simd_hint),
        DType::F64 => compare_elem_typed(lhs, rhs, out_dtype, f64_cmp),
        DType::F16 => {
            let scalar = f16::from_f64(rhs);
            compare_elem_typed(lhs, scalar, out_dtype, |a: f16, b: f16| {
                f32_cmp(a.to_f32(), b.to_f32())
            })
        }
        DType::BF16 => {
            let scalar = bf16::from_f64(rhs);
            compare_elem_typed(lhs, scalar, out_dtype, |a: bf16, b: bf16| {
                f32_cmp(a.to_f32(), b.to_f32())
            })
        }
        _ => panic!("compare_elem: unsupported dtype {:?}", dtype),
    }
}

/// Specialized scalar comparison for f32 with SIMD fast path.
#[cfg(feature = "simd")]
fn compare_elem_f32<Cmp>(
    lhs: FlexTensor,
    rhs: f32,
    out_dtype: BoolDType,
    cmp: Cmp,
    simd_hint: Option<CompareOp>,
) -> FlexTensor
where
    Cmp: Fn(f32, f32) -> bool,
{
    // SIMD fast path: tensor is contiguous
    if let Some((start, end)) = lhs.layout().contiguous_offsets()
        && let Some(simd_op) = simd_hint
    {
        let shape = lhs.layout().shape().clone();
        let lhs_storage: &[f32] = lhs.storage();
        let l_slice = &lhs_storage[start..end];

        let mut result = vec![0u8; l_slice.len()];
        simd::cmp_scalar_f32(l_slice, rhs, &mut result, simd_op);

        return make_bool_tensor(result, shape, out_dtype);
    }

    // Fallback to generic path
    compare_elem_typed(lhs, rhs, out_dtype, cmp)
}

/// Fallback when SIMD is disabled.
#[cfg(not(feature = "simd"))]
fn compare_elem_f32<Cmp>(
    lhs: FlexTensor,
    rhs: f32,
    out_dtype: BoolDType,
    cmp: Cmp,
    _simd_hint: Option<CompareOp>,
) -> FlexTensor
where
    Cmp: Fn(f32, f32) -> bool,
{
    compare_elem_typed(lhs, rhs, out_dtype, cmp)
}

fn compare_typed<E, Cmp>(
    lhs: FlexTensor,
    rhs: &FlexTensor,
    out_dtype: BoolDType,
    cmp: Cmp,
) -> FlexTensor
where
    E: Element + Pod,
    Cmp: Fn(E, E) -> bool,
{
    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[E] = lhs.storage();
    let rhs_storage: &[E] = rhs.storage();

    let result: Vec<u8> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        (Some((l_start, l_end)), Some((r_start, r_end))) => {
            let l_slice = &lhs_storage[l_start..l_end];
            let r_slice = &rhs_storage[r_start..r_end];
            l_slice
                .iter()
                .zip(r_slice)
                .map(|(&a, &b)| cmp(a, b) as u8)
                .collect()
        }
        // Fast path for 2D non-contiguous (common for transpose)
        _ if lhs.layout().num_dims() == 2 => crate::ops::binary::apply_2d_strided(
            lhs_storage,
            rhs_storage,
            lhs.layout(),
            rhs.layout(),
            |a, b| cmp(a, b) as u8,
        ),
        _ => {
            let lhs_iter = StridedIter::new(lhs.layout());
            let rhs_iter = StridedIter::new(rhs.layout());
            lhs_iter
                .zip(rhs_iter)
                .map(|(li, ri)| cmp(lhs_storage[li], rhs_storage[ri]) as u8)
                .collect()
        }
    };

    make_bool_tensor(result, shape, out_dtype)
}

fn compare_elem_typed<E, Cmp>(lhs: FlexTensor, rhs: E, out_dtype: BoolDType, cmp: Cmp) -> FlexTensor
where
    E: Element + Pod + Copy,
    Cmp: Fn(E, E) -> bool,
{
    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[E] = lhs.storage();

    let result: Vec<u8> = match lhs.layout().contiguous_offsets() {
        Some((start, end)) => lhs_storage[start..end]
            .iter()
            .map(|&a| cmp(a, rhs) as u8)
            .collect(),
        None => StridedIter::new(lhs.layout())
            .map(|idx| cmp(lhs_storage[idx], rhs) as u8)
            .collect(),
    };

    make_bool_tensor(result, shape, out_dtype)
}

/// Build a bool `FlexTensor` from a `Vec<u8>` of 0/1 bytes, tagged with the
/// requested output dtype.
///
/// burn-flex stores bools as 1 byte per element, so only Native and U8 are
/// supported. `Bool(U32)` would require 4-byte-per-element storage throughout
/// the backend; `dtype_usage` declares it unsupported and this function panics
/// if it's requested.
pub(crate) fn make_bool_tensor(data: Vec<u8>, shape: Shape, out_dtype: BoolDType) -> FlexTensor {
    let store = match out_dtype {
        BoolDType::Native => BoolStore::Native,
        BoolDType::U8 => BoolStore::U8,
        BoolDType::U32 => panic!(
            "burn-flex does not support Bool(U32) storage (only Native and U8). \
             Use a backend that declares Bool(U32) support, or work with Bool(Native)/Bool(U8)."
        ),
    };
    let bytes = Bytes::from_elems(data);
    FlexTensor::new(bytes, Layout::contiguous(shape), DType::Bool(store))
}

// Specific comparison functions

pub fn greater(lhs: FlexTensor, rhs: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    compare(
        lhs,
        rhs,
        out_dtype,
        |a, b| a > b,
        |a, b| a > b,
        Some(CompareOp::Gt),
    )
}

pub fn greater_elem(lhs: FlexTensor, rhs: f64, out_dtype: BoolDType) -> FlexTensor {
    compare_elem(
        lhs,
        rhs,
        out_dtype,
        |a, b| a > b,
        |a, b| a > b,
        Some(CompareOp::Gt),
    )
}

pub fn greater_equal(lhs: FlexTensor, rhs: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    compare(
        lhs,
        rhs,
        out_dtype,
        |a, b| a >= b,
        |a, b| a >= b,
        Some(CompareOp::Ge),
    )
}

pub fn greater_equal_elem(lhs: FlexTensor, rhs: f64, out_dtype: BoolDType) -> FlexTensor {
    compare_elem(
        lhs,
        rhs,
        out_dtype,
        |a, b| a >= b,
        |a, b| a >= b,
        Some(CompareOp::Ge),
    )
}

pub fn lower(lhs: FlexTensor, rhs: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    compare(
        lhs,
        rhs,
        out_dtype,
        |a, b| a < b,
        |a, b| a < b,
        Some(CompareOp::Lt),
    )
}

pub fn lower_elem(lhs: FlexTensor, rhs: f64, out_dtype: BoolDType) -> FlexTensor {
    compare_elem(
        lhs,
        rhs,
        out_dtype,
        |a, b| a < b,
        |a, b| a < b,
        Some(CompareOp::Lt),
    )
}

pub fn lower_equal(lhs: FlexTensor, rhs: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    compare(
        lhs,
        rhs,
        out_dtype,
        |a, b| a <= b,
        |a, b| a <= b,
        Some(CompareOp::Le),
    )
}

pub fn lower_equal_elem(lhs: FlexTensor, rhs: f64, out_dtype: BoolDType) -> FlexTensor {
    compare_elem(
        lhs,
        rhs,
        out_dtype,
        |a, b| a <= b,
        |a, b| a <= b,
        Some(CompareOp::Le),
    )
}

pub fn equal(lhs: FlexTensor, rhs: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    compare(
        lhs,
        rhs,
        out_dtype,
        |a, b| a == b,
        |a, b| a == b,
        Some(CompareOp::Eq),
    )
}

pub fn equal_elem(lhs: FlexTensor, rhs: f64, out_dtype: BoolDType) -> FlexTensor {
    compare_elem(
        lhs,
        rhs,
        out_dtype,
        |a, b| a == b,
        |a, b| a == b,
        Some(CompareOp::Eq),
    )
}

pub fn not_equal(lhs: FlexTensor, rhs: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    compare(
        lhs,
        rhs,
        out_dtype,
        |a, b| a != b,
        |a, b| a != b,
        Some(CompareOp::Ne),
    )
}

pub fn not_equal_elem(lhs: FlexTensor, rhs: f64, out_dtype: BoolDType) -> FlexTensor {
    compare_elem(
        lhs,
        rhs,
        out_dtype,
        |a, b| a != b,
        |a, b| a != b,
        Some(CompareOp::Ne),
    )
}

// Integer comparison functions

fn compare_int<I64Cmp, U64Cmp>(
    lhs: FlexTensor,
    rhs: FlexTensor,
    out_dtype: BoolDType,
    i64_cmp: I64Cmp,
    u64_cmp: U64Cmp,
) -> FlexTensor
where
    I64Cmp: Fn(i64, i64) -> bool,
    U64Cmp: Fn(u64, u64) -> bool,
{
    let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

    match lhs.dtype() {
        DType::I64 => compare_typed(lhs, &rhs, out_dtype, i64_cmp),
        DType::U64 => compare_typed(lhs, &rhs, out_dtype, u64_cmp),
        DType::I32 => compare_typed(lhs, &rhs, out_dtype, |a: i32, b: i32| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::I16 => compare_typed(lhs, &rhs, out_dtype, |a: i16, b: i16| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::I8 => compare_typed(lhs, &rhs, out_dtype, |a: i8, b: i8| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::U32 => compare_typed(lhs, &rhs, out_dtype, |a: u32, b: u32| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::U16 => compare_typed(lhs, &rhs, out_dtype, |a: u16, b: u16| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::U8 => compare_typed(lhs, &rhs, out_dtype, |a: u8, b: u8| {
            i64_cmp(a as i64, b as i64)
        }),
        other => panic!("compare_int: unsupported dtype {:?}", other),
    }
}

fn compare_int_elem<I64Cmp, U64Cmp>(
    lhs: FlexTensor,
    i64_rhs: i64,
    u64_rhs: u64,
    out_dtype: BoolDType,
    i64_cmp: I64Cmp,
    u64_cmp: U64Cmp,
) -> FlexTensor
where
    I64Cmp: Fn(i64, i64) -> bool,
    U64Cmp: Fn(u64, u64) -> bool,
{
    match lhs.dtype() {
        DType::I64 => compare_elem_typed(lhs, i64_rhs, out_dtype, i64_cmp),
        DType::U64 => compare_elem_typed(lhs, u64_rhs, out_dtype, u64_cmp),
        DType::I32 => compare_elem_typed(lhs, i64_rhs as i32, out_dtype, |a: i32, b: i32| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::I16 => compare_elem_typed(lhs, i64_rhs as i16, out_dtype, |a: i16, b: i16| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::I8 => compare_elem_typed(lhs, i64_rhs as i8, out_dtype, |a: i8, b: i8| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::U32 => compare_elem_typed(lhs, i64_rhs as u32, out_dtype, |a: u32, b: u32| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::U16 => compare_elem_typed(lhs, i64_rhs as u16, out_dtype, |a: u16, b: u16| {
            i64_cmp(a as i64, b as i64)
        }),
        DType::U8 => compare_elem_typed(lhs, i64_rhs as u8, out_dtype, |a: u8, b: u8| {
            i64_cmp(a as i64, b as i64)
        }),
        other => panic!("compare_int_elem: unsupported dtype {:?}", other),
    }
}

pub fn int_greater(lhs: FlexTensor, rhs: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    compare_int(lhs, rhs, out_dtype, |a, b| a > b, |a, b| a > b)
}

pub fn int_greater_elem(
    lhs: FlexTensor,
    i64_rhs: i64,
    u64_rhs: u64,
    out_dtype: BoolDType,
) -> FlexTensor {
    compare_int_elem(lhs, i64_rhs, u64_rhs, out_dtype, |a, b| a > b, |a, b| a > b)
}

pub fn int_greater_equal(lhs: FlexTensor, rhs: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    compare_int(lhs, rhs, out_dtype, |a, b| a >= b, |a, b| a >= b)
}

pub fn int_greater_equal_elem(
    lhs: FlexTensor,
    i64_rhs: i64,
    u64_rhs: u64,
    out_dtype: BoolDType,
) -> FlexTensor {
    compare_int_elem(
        lhs,
        i64_rhs,
        u64_rhs,
        out_dtype,
        |a, b| a >= b,
        |a, b| a >= b,
    )
}

pub fn int_lower(lhs: FlexTensor, rhs: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    compare_int(lhs, rhs, out_dtype, |a, b| a < b, |a, b| a < b)
}

pub fn int_lower_elem(
    lhs: FlexTensor,
    i64_rhs: i64,
    u64_rhs: u64,
    out_dtype: BoolDType,
) -> FlexTensor {
    compare_int_elem(lhs, i64_rhs, u64_rhs, out_dtype, |a, b| a < b, |a, b| a < b)
}

pub fn int_lower_equal(lhs: FlexTensor, rhs: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    compare_int(lhs, rhs, out_dtype, |a, b| a <= b, |a, b| a <= b)
}

pub fn int_lower_equal_elem(
    lhs: FlexTensor,
    i64_rhs: i64,
    u64_rhs: u64,
    out_dtype: BoolDType,
) -> FlexTensor {
    compare_int_elem(
        lhs,
        i64_rhs,
        u64_rhs,
        out_dtype,
        |a, b| a <= b,
        |a, b| a <= b,
    )
}

pub fn int_equal(lhs: FlexTensor, rhs: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    compare_int(lhs, rhs, out_dtype, |a, b| a == b, |a, b| a == b)
}

pub fn int_equal_elem(
    lhs: FlexTensor,
    i64_rhs: i64,
    u64_rhs: u64,
    out_dtype: BoolDType,
) -> FlexTensor {
    compare_int_elem(
        lhs,
        i64_rhs,
        u64_rhs,
        out_dtype,
        |a, b| a == b,
        |a, b| a == b,
    )
}

pub fn int_not_equal(lhs: FlexTensor, rhs: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    compare_int(lhs, rhs, out_dtype, |a, b| a != b, |a, b| a != b)
}

pub fn int_not_equal_elem(
    lhs: FlexTensor,
    i64_rhs: i64,
    u64_rhs: u64,
    out_dtype: BoolDType,
) -> FlexTensor {
    compare_int_elem(
        lhs,
        i64_rhs,
        u64_rhs,
        out_dtype,
        |a, b| a != b,
        |a, b| a != b,
    )
}

pub fn bool_not_equal(lhs: FlexTensor, rhs: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);
    let shape = lhs.layout().shape().clone();
    let lhs_data: &[u8] = lhs.bytes();
    let rhs_data: &[u8] = rhs.bytes();
    let result: Vec<u8> = match (
        lhs.layout().contiguous_offsets(),
        rhs.layout().contiguous_offsets(),
    ) {
        (Some((ls, le)), Some((rs, re))) => lhs_data[ls..le]
            .iter()
            .zip(&rhs_data[rs..re])
            .map(|(&a, &b)| if a != b { 1 } else { 0 })
            .collect(),
        _ => {
            let lhs = lhs.to_contiguous();
            let rhs = rhs.to_contiguous();
            lhs.bytes()
                .iter()
                .zip(rhs.bytes())
                .map(|(&a, &b)| if a != b { 1 } else { 0 })
                .collect()
        }
    };
    make_bool_tensor(result, shape, out_dtype)
}

pub fn bool_not_equal_elem(lhs: FlexTensor, rhs: bool, out_dtype: BoolDType) -> FlexTensor {
    let rhs_val: u8 = if rhs { 1 } else { 0 };
    let shape = lhs.layout().shape().clone();
    let lhs = lhs.to_contiguous();
    let data: &[u8] = lhs.bytes();
    let result: Vec<u8> = data
        .iter()
        .map(|&a| if a != rhs_val { 1 } else { 0 })
        .collect();
    make_bool_tensor(result, shape, out_dtype)
}

// ============================================================================
// any / all operations
// ============================================================================

/// Check if any element is non-zero (float tensors).
pub fn any_float(tensor: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    let has_any = match tensor.dtype() {
        DType::F32 => iter_elements::<f32>(&tensor).any(|x| x != 0.0),
        DType::F64 => iter_elements::<f64>(&tensor).any(|x| x != 0.0),
        DType::F16 => iter_elements::<f16>(&tensor).any(|x: f16| x.to_f32() != 0.0),
        DType::BF16 => iter_elements::<bf16>(&tensor).any(|x: bf16| x.to_f32() != 0.0),
        _ => panic!("any_float: unsupported dtype {:?}", tensor.dtype()),
    };
    bool_scalar(has_any, out_dtype)
}

/// Check if any element along a dimension is non-zero (float tensors).
pub fn any_float_dim(tensor: FlexTensor, dim: usize, out_dtype: BoolDType) -> FlexTensor {
    reduce_bool_dim(&tensor, dim, false, |a, b| a || b, out_dtype)
}

/// Check if all elements are non-zero (float tensors).
pub fn all_float(tensor: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    let all = match tensor.dtype() {
        DType::F32 => iter_elements::<f32>(&tensor).all(|x| x != 0.0),
        DType::F64 => iter_elements::<f64>(&tensor).all(|x| x != 0.0),
        DType::F16 => iter_elements::<f16>(&tensor).all(|x: f16| x.to_f32() != 0.0),
        DType::BF16 => iter_elements::<bf16>(&tensor).all(|x: bf16| x.to_f32() != 0.0),
        _ => panic!("all_float: unsupported dtype {:?}", tensor.dtype()),
    };
    bool_scalar(all, out_dtype)
}

/// Check if all elements along a dimension are non-zero (float tensors).
pub fn all_float_dim(tensor: FlexTensor, dim: usize, out_dtype: BoolDType) -> FlexTensor {
    reduce_bool_dim(&tensor, dim, true, |a, b| a && b, out_dtype)
}

/// Check if any element is non-zero (int tensors).
pub fn any_int(tensor: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    let has_any = match tensor.dtype() {
        DType::I64 => iter_elements::<i64>(&tensor).any(|x| x != 0),
        DType::I32 => iter_elements::<i32>(&tensor).any(|x| x != 0),
        DType::I16 => iter_elements::<i16>(&tensor).any(|x| x != 0),
        DType::I8 => iter_elements::<i8>(&tensor).any(|x| x != 0),
        DType::U64 => iter_elements::<u64>(&tensor).any(|x| x != 0),
        DType::U32 => iter_elements::<u32>(&tensor).any(|x| x != 0),
        DType::U16 => iter_elements::<u16>(&tensor).any(|x| x != 0),
        DType::U8 => iter_elements::<u8>(&tensor).any(|x| x != 0),
        _ => panic!("any_int: unsupported dtype {:?}", tensor.dtype()),
    };
    bool_scalar(has_any, out_dtype)
}

/// Check if any element along a dimension is non-zero (int tensors).
pub fn any_int_dim(tensor: FlexTensor, dim: usize, out_dtype: BoolDType) -> FlexTensor {
    reduce_bool_dim_int(&tensor, dim, false, |a, b| a || b, out_dtype)
}

/// Check if all elements are non-zero (int tensors).
pub fn all_int(tensor: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    let all = match tensor.dtype() {
        DType::I64 => iter_elements::<i64>(&tensor).all(|x| x != 0),
        DType::I32 => iter_elements::<i32>(&tensor).all(|x| x != 0),
        DType::I16 => iter_elements::<i16>(&tensor).all(|x| x != 0),
        DType::I8 => iter_elements::<i8>(&tensor).all(|x| x != 0),
        DType::U64 => iter_elements::<u64>(&tensor).all(|x| x != 0),
        DType::U32 => iter_elements::<u32>(&tensor).all(|x| x != 0),
        DType::U16 => iter_elements::<u16>(&tensor).all(|x| x != 0),
        DType::U8 => iter_elements::<u8>(&tensor).all(|x| x != 0),
        _ => panic!("all_int: unsupported dtype {:?}", tensor.dtype()),
    };
    bool_scalar(all, out_dtype)
}

/// Check if all elements along a dimension are non-zero (int tensors).
pub fn all_int_dim(tensor: FlexTensor, dim: usize, out_dtype: BoolDType) -> FlexTensor {
    reduce_bool_dim_int(&tensor, dim, true, |a, b| a && b, out_dtype)
}

/// Check if any bool element is true.
pub fn any_bool(tensor: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let data: &[u8] = tensor.bytes();
    bool_scalar(data.iter().any(|&x| x != 0), out_dtype)
}

/// Check if any bool element along a dimension is true.
pub fn any_bool_dim(tensor: FlexTensor, dim: usize, out_dtype: BoolDType) -> FlexTensor {
    reduce_bool_dim_raw(&tensor, dim, false, |a, b| a || b, out_dtype)
}

/// Check if all bool elements are true.
pub fn all_bool(tensor: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let data: &[u8] = tensor.bytes();
    bool_scalar(data.iter().all(|&x| x != 0), out_dtype)
}

/// Check if all bool elements along a dimension are true.
pub fn all_bool_dim(tensor: FlexTensor, dim: usize, out_dtype: BoolDType) -> FlexTensor {
    reduce_bool_dim_raw(&tensor, dim, true, |a, b| a && b, out_dtype)
}

// ============================================================================
// Helpers for any/all
// ============================================================================

fn bool_scalar(val: bool, out_dtype: BoolDType) -> FlexTensor {
    let byte: u8 = if val { 1 } else { 0 };
    make_bool_tensor(alloc::vec![byte], Shape::from(alloc::vec![1]), out_dtype)
}

fn iter_elements<'a, E: Element + Pod + 'a>(
    tensor: &'a FlexTensor,
) -> Box<dyn Iterator<Item = E> + 'a> {
    let data: &[E] = tensor.storage();
    match tensor.layout().contiguous_offsets() {
        Some((start, end)) => Box::new(data[start..end].iter().copied()),
        None => Box::new(StridedIter::new(tensor.layout()).map(move |idx| data[idx])),
    }
}

/// Reduce along a dimension producing a bool tensor.
///
/// The `is_nonzero` closure reads the data slice at a given index and returns
/// whether the element is nonzero.
fn reduce_bool_dim_with(
    tensor: &FlexTensor,
    dim: usize,
    init: bool,
    combine: fn(bool, bool) -> bool,
    out_dtype: BoolDType,
    is_nonzero: impl Fn(usize) -> bool,
) -> FlexTensor {
    debug_assert!(tensor.is_contiguous() && tensor.layout().start_offset() == 0);
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let out_size = outer_size.max(1) * inner_size.max(1);
    let mut result: Vec<u8> = Vec::with_capacity(out_size);

    for outer in 0..outer_size.max(1) {
        for inner in 0..inner_size.max(1) {
            let mut acc = init;
            for d in 0..dim_size {
                let idx = outer * dim_size * inner_size + d * inner_size + inner;
                acc = combine(acc, is_nonzero(idx));
            }
            result.push(if acc { 1 } else { 0 });
        }
    }

    make_bool_tensor(result, Shape::from(out_shape), out_dtype)
}

/// Reduce along a dimension producing a bool tensor (for float any/all_dim).
fn reduce_bool_dim(
    tensor: &FlexTensor,
    dim: usize,
    init: bool,
    combine: fn(bool, bool) -> bool,
    out_dtype: BoolDType,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    match tensor.dtype() {
        DType::F32 => {
            let data: &[f32] = tensor.storage();
            reduce_bool_dim_with(&tensor, dim, init, combine, out_dtype, |idx| {
                data[idx] != 0.0
            })
        }
        DType::F64 => {
            let data: &[f64] = tensor.storage();
            reduce_bool_dim_with(&tensor, dim, init, combine, out_dtype, |idx| {
                data[idx] != 0.0
            })
        }
        DType::F16 => {
            let data: &[f16] = tensor.storage();
            reduce_bool_dim_with(&tensor, dim, init, combine, out_dtype, |idx| {
                data[idx].to_f32() != 0.0
            })
        }
        DType::BF16 => {
            let data: &[bf16] = tensor.storage();
            reduce_bool_dim_with(&tensor, dim, init, combine, out_dtype, |idx| {
                data[idx].to_f32() != 0.0
            })
        }
        _ => panic!("reduce_bool_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Reduce along a dimension producing a bool tensor (for int any/all_dim).
fn reduce_bool_dim_int(
    tensor: &FlexTensor,
    dim: usize,
    init: bool,
    combine: fn(bool, bool) -> bool,
    out_dtype: BoolDType,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    macro_rules! dispatch {
        ($ty:ty) => {{
            let data: &[$ty] = tensor.storage();
            reduce_bool_dim_with(&tensor, dim, init, combine, out_dtype, |idx| data[idx] != 0)
        }};
    }
    match tensor.dtype() {
        DType::I64 => dispatch!(i64),
        DType::I32 => dispatch!(i32),
        DType::I16 => dispatch!(i16),
        DType::I8 => dispatch!(i8),
        DType::U64 => dispatch!(u64),
        DType::U32 => dispatch!(u32),
        DType::U16 => dispatch!(u16),
        DType::U8 => dispatch!(u8),
        other => panic!("reduce_bool_dim_int: unsupported dtype {:?}", other),
    }
}

/// Reduce along a dimension producing a bool tensor (for bool any/all_dim).
fn reduce_bool_dim_raw(
    tensor: &FlexTensor,
    dim: usize,
    init: bool,
    combine: fn(bool, bool) -> bool,
    out_dtype: BoolDType,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let data: &[u8] = tensor.bytes();
    reduce_bool_dim_with(&tensor, dim, init, combine, out_dtype, |idx| data[idx] != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_greater() {
        let lhs = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]));
        let rhs = FlexTensor::from_data(TensorData::new(vec![2.0f32, 2.0, 1.0], [3]));
        let result = greater(lhs, rhs, BoolDType::Native);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[0, 0, 1]); // 1>2=F, 2>2=F, 3>1=T
    }

    #[test]
    fn test_greater_elem() {
        let lhs = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]));
        let result = greater_elem(lhs, 2.0, BoolDType::Native);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[0, 0, 1]); // 1>2=F, 2>2=F, 3>2=T
    }

    #[test]
    fn test_equal() {
        let lhs = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]));
        let rhs = FlexTensor::from_data(TensorData::new(vec![1.0f32, 3.0, 3.0], [3]));
        let result = equal(lhs, rhs, BoolDType::Native);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[1, 0, 1]); // 1==1=T, 2==3=F, 3==3=T
    }

    // === Non-contiguous tensor tests ===

    fn tensor_2d(data: Vec<f32>, rows: usize, cols: usize) -> FlexTensor {
        FlexTensor::from_data(TensorData::new(data, vec![rows, cols]))
    }

    #[test]
    fn test_greater_transposed() {
        // [[1, 2], [3, 4]] transposed -> [[1, 3], [2, 4]]
        // Compare with [[2, 2], [2, 2]]
        // [[1, 3], [2, 4]] > [[2, 2], [2, 2]] = [[F, T], [F, T]]
        let lhs = tensor_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let lhs = lhs.transpose(0, 1);
        assert!(!lhs.is_contiguous());

        let rhs = tensor_2d(vec![2.0, 2.0, 2.0, 2.0], 2, 2);
        let result = greater(lhs, rhs, BoolDType::Native);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[0, 1, 0, 1]);
    }

    #[test]
    fn test_equal_flipped_1d() {
        // [1, 2, 3, 4] flipped -> [4, 3, 2, 1]
        // Compare [4, 3, 2, 1] == [4, 2, 2, 1] = [T, F, T, T]
        let lhs = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let lhs = crate::ops::flip::flip(lhs, &[0]);
        assert!(lhs.layout().strides()[0] < 0);

        let rhs = FlexTensor::from_data(TensorData::new(vec![4.0f32, 2.0, 2.0, 1.0], [4]));
        let result = equal(lhs, rhs, BoolDType::Native);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[1, 0, 1, 1]);
    }

    #[test]
    fn test_lower_flipped_2d() {
        // [[1, 2], [3, 4]] with axis 0 flipped -> [[3, 4], [1, 2]]
        // [[3, 4], [1, 2]] < [[2, 5], [2, 1]] = [[F, T], [T, F]]
        let lhs = tensor_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let lhs = crate::ops::flip::flip(lhs, &[0]);
        assert!(lhs.layout().strides()[0] < 0);

        let rhs = tensor_2d(vec![2.0, 5.0, 2.0, 1.0], 2, 2);
        let result = lower(lhs, rhs, BoolDType::Native);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[0, 1, 1, 0]);
    }

    #[test]
    fn test_greater_elem_flipped() {
        // [1, 2, 3, 4] flipped -> [4, 3, 2, 1]
        // [4, 3, 2, 1] > 2.5 = [T, T, F, F]
        let lhs = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let lhs = crate::ops::flip::flip(lhs, &[0]);
        assert!(lhs.layout().strides()[0] < 0);

        let result = greater_elem(lhs, 2.5, BoolDType::Native);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[1, 1, 0, 0]);
    }

    #[test]
    fn test_equal_both_transposed() {
        // Both tensors transposed
        // [[1, 2], [3, 4]]^T -> [[1, 3], [2, 4]]
        // [[1, 3], [2, 4]]^T -> [[1, 2], [3, 4]]
        // [[1, 3], [2, 4]] == [[1, 2], [3, 4]] = [[T, F], [F, T]]
        let lhs = tensor_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2).transpose(0, 1);
        let rhs = tensor_2d(vec![1.0, 3.0, 2.0, 4.0], 2, 2).transpose(0, 1);
        assert!(!lhs.is_contiguous());
        assert!(!rhs.is_contiguous());

        let result = equal(lhs, rhs, BoolDType::Native);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[1, 0, 0, 1]);
    }

    #[test]
    fn test_not_equal_narrowed() {
        // [1, 2, 3, 4, 5, 6] narrowed to [2, 3, 4, 5]
        // [2, 3, 4, 5] != [2, 2, 4, 4] = [F, T, F, T]
        let lhs =
            FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [6]));
        let lhs = lhs.narrow(0, 1, 4);

        let rhs = FlexTensor::from_data(TensorData::new(vec![2.0f32, 2.0, 4.0, 4.0], [4]));
        let result = not_equal(lhs, rhs, BoolDType::Native);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[0, 1, 0, 1]);
    }

    #[test]
    fn test_int_greater_flipped() {
        // Integer comparison with flipped tensor
        let lhs = FlexTensor::from_data(TensorData::new(vec![1i64, 2, 3, 4], [4]));
        let lhs = crate::ops::flip::flip(lhs, &[0]);
        assert!(lhs.layout().strides()[0] < 0);

        // [4, 3, 2, 1] > [3, 3, 3, 3] = [T, F, F, F]
        let rhs = FlexTensor::from_data(TensorData::new(vec![3i64, 3, 3, 3], [4]));
        let result = int_greater(lhs, rhs, BoolDType::Native);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[1, 0, 0, 0]);
    }

    #[test]
    fn test_lower_flipped_both_axes() {
        // [[1, 2], [3, 4]] flipped on both axes -> [[4, 3], [2, 1]]
        // [[4, 3], [2, 1]] < [[3, 3], [3, 3]] = [[F, F], [T, T]]
        let lhs = tensor_2d(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let lhs = crate::ops::flip::flip(lhs, &[0, 1]);
        assert!(lhs.layout().strides()[0] < 0);
        assert!(lhs.layout().strides()[1] < 0);

        let rhs = tensor_2d(vec![3.0, 3.0, 3.0, 3.0], 2, 2);
        let result = lower(lhs, rhs, BoolDType::Native);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[0, 0, 1, 1]);
    }

    #[test]
    fn test_any_float_dim_transposed() {
        // [[0, 1], [0, 0]] transposed -> [[0, 0], [1, 0]]
        // any_dim along dim 1: [false, true] -> [0, 1]
        // Before the C2 fix, to_contiguous was called inside reduce_bool_dim_with
        // AFTER taking a reference to the original non-contiguous storage,
        // causing stale pointer reads.
        let tensor = tensor_2d(vec![0.0, 1.0, 0.0, 0.0], 2, 2);
        let transposed = tensor.transpose(0, 1);
        assert!(!transposed.is_contiguous());

        let result = any_float_dim(transposed, 1, BoolDType::Native);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[0, 1]); // row 0: all zeros; row 1: has a 1
    }

    #[test]
    fn test_any_float_dim_narrowed() {
        // [0, 5, 0, 3, 0, 0] narrowed to [[5, 0], [3, 0]] (shape [2,2])
        // any_dim along dim 1: [true, true] -> [1, 1]
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![0.0f32, 5.0, 0.0, 3.0, 0.0, 0.0],
            [3, 2],
        ));
        let narrowed = tensor.narrow(0, 0, 2); // first 2 rows: [[0, 5], [0, 3]]
        let result = any_float_dim(narrowed, 1, BoolDType::Native);
        let data: &[u8] = result.bytes();
        assert_eq!(data, &[1, 1]);
    }
}
