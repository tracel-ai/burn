//! Reduction operations for FlexTensor.
//!
//! Optimized with:
//! - Strided iteration (no copy for non-contiguous tensors)
//! - Portable SIMD via macerator (NEON, AVX2, SIMD128, scalar fallback)
//! - Rayon parallelism for large tensors

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, bf16, f16};

use crate::strided_index::StridedIter;
use crate::{FlexTensor, Layout};

use super::{INDEX_DTYPE, float_storage_as_f32};

/// Assert that a dimension size fits in `isize`, which is required for index-producing
/// operations (argmax, argmin, *_with_indices) that store dimension indices as `isize`.
#[inline(always)]
fn assert_dim_fits_isize(dim_size: usize, dim: usize) {
    assert!(
        dim_size <= isize::MAX as usize,
        "dimension {dim} has size {dim_size} which exceeds isize::MAX"
    );
}

#[cfg(feature = "simd")]
use crate::simd::kernels;

#[cfg(feature = "simd")]
use crate::simd::aligned;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Truncate an i64 to a smaller Pod type, keeping the low-order bytes.
/// Endian-safe: works correctly on both little-endian and big-endian targets.
fn truncate_i64_to_pod<E: bytemuck::Pod>(value: i64) -> E {
    let bytes = value.to_ne_bytes();
    let size = core::mem::size_of::<E>();
    debug_assert!(size <= core::mem::size_of::<i64>());
    let offset = if cfg!(target_endian = "big") {
        core::mem::size_of::<i64>() - size
    } else {
        0
    };
    bytemuck::pod_read_unaligned(&bytes[offset..offset + size])
}

// ============================================================================
// Sum (all elements)
// ============================================================================

/// Sum all elements in a tensor, returning a scalar tensor.
pub fn sum(tensor: FlexTensor) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => sum_f32(&tensor),
        DType::F64 => sum_impl::<f64>(&tensor),
        DType::F16 => reduce_scalar_half(&tensor, |a, b| a + b, 0.0, f16::to_f32, f16::from_f32),
        DType::BF16 => reduce_scalar_half(&tensor, |a, b| a + b, 0.0, bf16::to_f32, bf16::from_f32),
        DType::I8 => sum_impl_widening::<i8>(&tensor),
        DType::I16 => sum_impl_widening::<i16>(&tensor),
        DType::I32 => sum_impl_widening::<i32>(&tensor),
        DType::I64 => sum_impl::<i64>(&tensor),
        DType::U8 => sum_impl_widening::<u8>(&tensor),
        DType::U16 => sum_impl_widening::<u16>(&tensor),
        DType::U32 => sum_impl_widening::<u32>(&tensor),
        DType::U64 => sum_impl::<u64>(&tensor),
        _ => panic!("sum: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Optimized f32 sum with SIMD and parallelism.
fn sum_f32(tensor: &FlexTensor) -> FlexTensor {
    let result = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[f32] = tensor.storage();
            let slice = &data[start..end];
            sum_f32_contiguous(slice)
        }
        None => {
            // Non-contiguous: check if we can sum the buffer directly.
            // For transposed tensors that use all elements (no slicing),
            // the sum is the same regardless of element order.
            let data: &[f32] = tensor.storage();
            let elem_count = tensor.layout().num_elements();

            if data.len() == elem_count {
                // Tensor uses entire buffer - sum directly (order doesn't matter for sum)
                sum_f32_contiguous(data)
            } else {
                // Sliced or partial view - must use strided iteration
                StridedIter::new(tensor.layout()).map(|idx| data[idx]).sum()
            }
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    FlexTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::F32)
}

/// SIMD + parallel sum for contiguous f32 slice.
///
/// The parallel threshold is higher than the general `PARALLEL_THRESHOLD` because
/// sum is memory-bound and L2-resident data (< ~16 MiB / 4M f32 elements on
/// Apple M-series) doesn't benefit from rayon's task dispatch overhead.
#[inline]
fn sum_f32_contiguous(data: &[f32]) -> f32 {
    #[cfg(feature = "rayon")]
    if data.len() >= 4 * 1024 * 1024 {
        return sum_f32_parallel(data);
    }

    #[cfg(feature = "simd")]
    {
        kernels::sum_f32(data)
    }

    #[cfg(not(feature = "simd"))]
    {
        data.iter().copied().sum()
    }
}

/// Parallel sum using rayon with SIMD per chunk.
#[cfg(feature = "rayon")]
#[inline]
fn sum_f32_parallel(data: &[f32]) -> f32 {
    const CHUNK_SIZE: usize = 64 * 1024; // 64K elements per chunk

    data.par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            #[cfg(feature = "simd")]
            {
                kernels::sum_f32(chunk)
            }
            #[cfg(not(feature = "simd"))]
            {
                chunk.iter().copied().sum::<f32>()
            }
        })
        .sum()
}

fn sum_impl<E: Element + bytemuck::Pod + Default + core::iter::Sum>(
    tensor: &FlexTensor,
) -> FlexTensor {
    let result: E = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[E] = tensor.storage();
            data[start..end].iter().copied().sum()
        }
        None => {
            let data: &[E] = tensor.storage();
            StridedIter::new(tensor.layout()).map(|idx| data[idx]).sum()
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(vec![1])),
        tensor.dtype(),
    )
}

/// Widening scalar reduction for small integer types: accumulate in i64 to avoid overflow.
macro_rules! widening_scalar_reduce {
    ($name:ident, $fold:expr, $init:expr) => {
        fn $name<E>(tensor: &FlexTensor) -> FlexTensor
        where
            E: Element + bytemuck::Pod + Default,
            i64: From<E>,
        {
            let total: i64 = match tensor.layout().contiguous_offsets() {
                Some((start, end)) => {
                    let data: &[E] = tensor.storage();
                    data[start..end]
                        .iter()
                        .fold($init, |acc, x| ($fold)(acc, i64::from(*x)))
                }
                None => {
                    let data: &[E] = tensor.storage();
                    StridedIter::new(tensor.layout())
                        .fold($init, |acc, idx| ($fold)(acc, i64::from(data[idx])))
                }
            };
            // Truncate back to target type (wrapping, matches PyTorch)
            let result: E = truncate_i64_to_pod(total);
            let bytes = Bytes::from_elems(vec![result]);
            FlexTensor::new(
                bytes,
                Layout::contiguous(Shape::from(vec![1])),
                tensor.dtype(),
            )
        }
    };
}

widening_scalar_reduce!(
    sum_impl_widening,
    |acc: i64, x: i64| acc.wrapping_add(x),
    0i64
);
widening_scalar_reduce!(
    prod_impl_widening,
    |acc: i64, x: i64| acc.wrapping_mul(x),
    1i64
);

/// Scalar reduction for half-precision types, accumulating in f32.
fn reduce_scalar_half<E>(
    tensor: &FlexTensor,
    fold: fn(f32, f32) -> f32,
    init: f32,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> FlexTensor
where
    E: Element + bytemuck::Pod,
{
    let result: f32 = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[E] = tensor.storage();
            data[start..end]
                .iter()
                .fold(init, |acc, x| fold(acc, to_f32(*x)))
        }
        None => {
            let data: &[E] = tensor.storage();
            StridedIter::new(tensor.layout()).fold(init, |acc, idx| fold(acc, to_f32(data[idx])))
        }
    };

    let bytes = Bytes::from_elems(vec![from_f32(result)]);
    FlexTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), E::dtype())
}

// ============================================================================
// Sum along dimension
// ============================================================================

/// Sum along a dimension, keeping the dimension with size 1.
pub fn sum_dim(tensor: FlexTensor, dim: usize) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => reduce_dim_f32(&tensor, dim, ReduceOp::Sum),
        DType::F64 => reduce_dim_impl::<f64, _>(&tensor, dim, 0.0, |acc, x| acc + x),
        DType::F16 => reduce_dim_half(
            &tensor,
            dim,
            0.0,
            |acc, x| acc + x,
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => reduce_dim_half(
            &tensor,
            dim,
            0.0,
            |acc, x| acc + x,
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I8 => reduce_dim_widening::<i8, _>(&tensor, dim, 0, |acc, x| acc.wrapping_add(x)),
        DType::I16 => reduce_dim_widening::<i16, _>(&tensor, dim, 0, |acc, x| acc.wrapping_add(x)),
        DType::I32 => reduce_dim_widening::<i32, _>(&tensor, dim, 0, |acc, x| acc.wrapping_add(x)),
        DType::I64 => reduce_dim_impl::<i64, _>(&tensor, dim, 0, |acc, x| acc + x),
        DType::U8 => reduce_dim_widening::<u8, _>(&tensor, dim, 0, |acc, x| acc.wrapping_add(x)),
        DType::U16 => reduce_dim_widening::<u16, _>(&tensor, dim, 0, |acc, x| acc.wrapping_add(x)),
        DType::U32 => reduce_dim_widening::<u32, _>(&tensor, dim, 0, |acc, x| acc.wrapping_add(x)),
        DType::U64 => reduce_dim_impl::<u64, _>(&tensor, dim, 0, |acc, x| acc + x),
        _ => panic!("sum_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Mean along a dimension, keeping the dimension with size 1.
pub fn mean_dim(tensor: FlexTensor, dim: usize) -> FlexTensor {
    let dim_size = tensor.layout().shape()[dim];
    assert!(
        dim_size > 0,
        "mean_dim: cannot take mean of empty dimension"
    );
    let dtype = tensor.dtype();

    // Half-precision types fuse sum+divide in f32 to avoid overflow when the
    // intermediate sum exceeds f16::MAX, so they don't go through sum_dim.
    match dtype {
        DType::F16 => return mean_dim_half::<f16>(&tensor, dim),
        DType::BF16 => return mean_dim_half::<bf16>(&tensor, dim),
        _ => {}
    }

    let sum_result = sum_dim(tensor, dim);

    // Divide by dimension size
    match dtype {
        DType::F32 => scalar_div::<f32>(sum_result, dim_size as f32),
        DType::F64 => scalar_div::<f64>(sum_result, dim_size as f64),
        DType::I8 => {
            let divisor = dim_size as i32;
            let mut tensor = sum_result;
            let data: &mut [i8] = tensor.storage_mut();
            for x in data.iter_mut() {
                *x = ((*x as i32) / divisor) as i8;
            }
            tensor
        }
        DType::I16 => {
            let divisor = dim_size as i32;
            let mut tensor = sum_result;
            let data: &mut [i16] = tensor.storage_mut();
            for x in data.iter_mut() {
                *x = ((*x as i32) / divisor) as i16;
            }
            tensor
        }
        DType::I32 => scalar_div::<i32>(sum_result, dim_size as i32),
        DType::I64 => scalar_div::<i64>(sum_result, dim_size as i64),
        DType::U8 => {
            let divisor = dim_size as u32;
            let mut tensor = sum_result;
            let data: &mut [u8] = tensor.storage_mut();
            for x in data.iter_mut() {
                *x = ((*x as u32) / divisor) as u8;
            }
            tensor
        }
        DType::U16 => {
            let divisor = dim_size as u32;
            let mut tensor = sum_result;
            let data: &mut [u16] = tensor.storage_mut();
            for x in data.iter_mut() {
                *x = ((*x as u32) / divisor) as u16;
            }
            tensor
        }
        DType::U32 => scalar_div::<u32>(sum_result, dim_size as u32),
        DType::U64 => scalar_div::<u64>(sum_result, dim_size as u64),
        _ => panic!("mean_dim: unsupported dtype {:?}", dtype),
    }
}

/// Product of all elements in a tensor, returning a scalar tensor.
pub fn prod(tensor: FlexTensor) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => prod_impl::<f32>(&tensor),
        DType::F64 => prod_impl::<f64>(&tensor),
        DType::F16 => reduce_scalar_half(&tensor, |a, b| a * b, 1.0, f16::to_f32, f16::from_f32),
        DType::BF16 => reduce_scalar_half(&tensor, |a, b| a * b, 1.0, bf16::to_f32, bf16::from_f32),
        DType::I8 => prod_impl_widening::<i8>(&tensor),
        DType::I16 => prod_impl_widening::<i16>(&tensor),
        DType::I32 => prod_impl_widening::<i32>(&tensor),
        DType::I64 => prod_impl::<i64>(&tensor),
        DType::U8 => prod_impl_widening::<u8>(&tensor),
        DType::U16 => prod_impl_widening::<u16>(&tensor),
        DType::U32 => prod_impl_widening::<u32>(&tensor),
        DType::U64 => prod_impl::<u64>(&tensor),
        _ => panic!("prod: unsupported dtype {:?}", tensor.dtype()),
    }
}

fn prod_impl<E: Element + bytemuck::Pod + Default + core::iter::Product>(
    tensor: &FlexTensor,
) -> FlexTensor {
    let result: E = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[E] = tensor.storage();
            data[start..end].iter().copied().product()
        }
        None => {
            let data: &[E] = tensor.storage();
            StridedIter::new(tensor.layout())
                .map(|idx| data[idx])
                .product()
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(vec![1])),
        tensor.dtype(),
    )
}

/// Product along a dimension, keeping the dimension with size 1.
pub fn prod_dim(tensor: FlexTensor, dim: usize) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => reduce_dim_f32(&tensor, dim, ReduceOp::Prod),
        DType::F64 => reduce_dim_impl::<f64, _>(&tensor, dim, 1.0, |acc, x| acc * x),
        DType::F16 => reduce_dim_half(
            &tensor,
            dim,
            1.0,
            |acc, x| acc * x,
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => reduce_dim_half(
            &tensor,
            dim,
            1.0,
            |acc, x| acc * x,
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I8 => reduce_dim_widening::<i8, _>(&tensor, dim, 1, |acc, x| acc.wrapping_mul(x)),
        DType::I16 => reduce_dim_widening::<i16, _>(&tensor, dim, 1, |acc, x| acc.wrapping_mul(x)),
        DType::I32 => reduce_dim_widening::<i32, _>(&tensor, dim, 1, |acc, x| acc.wrapping_mul(x)),
        DType::I64 => reduce_dim_impl::<i64, _>(&tensor, dim, 1, |acc, x| acc * x),
        DType::U8 => reduce_dim_widening::<u8, _>(&tensor, dim, 1, |acc, x| acc.wrapping_mul(x)),
        DType::U16 => reduce_dim_widening::<u16, _>(&tensor, dim, 1, |acc, x| acc.wrapping_mul(x)),
        DType::U32 => reduce_dim_widening::<u32, _>(&tensor, dim, 1, |acc, x| acc.wrapping_mul(x)),
        DType::U64 => reduce_dim_impl::<u64, _>(&tensor, dim, 1, |acc, x| acc * x),
        _ => panic!("prod_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

// ============================================================================
// Max / Min (all elements)
// ============================================================================

/// Max of all elements, returning a scalar tensor of shape \[1\].
pub fn max(tensor: FlexTensor) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => max_f32_reduce(&tensor),
        DType::F64 => max_impl::<f64>(&tensor),
        DType::F16 => reduce_scalar_half(
            &tensor,
            f32::max,
            f32::NEG_INFINITY,
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => reduce_scalar_half(
            &tensor,
            f32::max,
            f32::NEG_INFINITY,
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I8 => max_impl::<i8>(&tensor),
        DType::I16 => max_impl::<i16>(&tensor),
        DType::I32 => max_impl::<i32>(&tensor),
        DType::I64 => max_impl::<i64>(&tensor),
        DType::U8 => max_impl::<u8>(&tensor),
        DType::U16 => max_impl::<u16>(&tensor),
        DType::U32 => max_impl::<u32>(&tensor),
        DType::U64 => max_impl::<u64>(&tensor),
        _ => panic!("max: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Min of all elements, returning a scalar tensor of shape \[1\].
pub fn min(tensor: FlexTensor) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => min_f32_reduce(&tensor),
        DType::F64 => min_impl::<f64>(&tensor),
        DType::F16 => {
            reduce_scalar_half(&tensor, f32::min, f32::INFINITY, f16::to_f32, f16::from_f32)
        }
        DType::BF16 => reduce_scalar_half(
            &tensor,
            f32::min,
            f32::INFINITY,
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I8 => min_impl::<i8>(&tensor),
        DType::I16 => min_impl::<i16>(&tensor),
        DType::I32 => min_impl::<i32>(&tensor),
        DType::I64 => min_impl::<i64>(&tensor),
        DType::U8 => min_impl::<u8>(&tensor),
        DType::U16 => min_impl::<u16>(&tensor),
        DType::U32 => min_impl::<u32>(&tensor),
        DType::U64 => min_impl::<u64>(&tensor),
        _ => panic!("min: unsupported dtype {:?}", tensor.dtype()),
    }
}

fn max_f32_reduce(tensor: &FlexTensor) -> FlexTensor {
    let result = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[f32] = tensor.storage();
            max_f32_contiguous(&data[start..end])
        }
        None => {
            let data: &[f32] = tensor.storage();
            let elem_count = tensor.layout().num_elements();
            if data.len() == elem_count {
                // Non-contiguous but uses all elements (e.g., transposed)
                max_f32_contiguous(data)
            } else {
                StridedIter::new(tensor.layout())
                    .map(|idx| data[idx])
                    .reduce(|a, b| if a >= b { a } else { b })
                    .expect("max: tensor must not be empty")
            }
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    FlexTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::F32)
}

#[inline]
fn max_f32_contiguous(data: &[f32]) -> f32 {
    #[cfg(feature = "simd")]
    {
        kernels::max_f32(data)
    }

    #[cfg(not(feature = "simd"))]
    {
        data.iter()
            .copied()
            .reduce(|a, b| if a >= b { a } else { b })
            .expect("max: tensor must not be empty")
    }
}

fn min_f32_reduce(tensor: &FlexTensor) -> FlexTensor {
    let result = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[f32] = tensor.storage();
            min_f32_contiguous(&data[start..end])
        }
        None => {
            let data: &[f32] = tensor.storage();
            let elem_count = tensor.layout().num_elements();
            if data.len() == elem_count {
                min_f32_contiguous(data)
            } else {
                StridedIter::new(tensor.layout())
                    .map(|idx| data[idx])
                    .reduce(|a, b| if a <= b { a } else { b })
                    .expect("min: tensor must not be empty")
            }
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    FlexTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), DType::F32)
}

#[inline]
fn min_f32_contiguous(data: &[f32]) -> f32 {
    #[cfg(feature = "simd")]
    {
        kernels::min_f32(data)
    }

    #[cfg(not(feature = "simd"))]
    {
        data.iter()
            .copied()
            .reduce(|a, b| if a <= b { a } else { b })
            .expect("min: tensor must not be empty")
    }
}

fn max_impl<E: Element + bytemuck::Pod + PartialOrd>(tensor: &FlexTensor) -> FlexTensor {
    let result: E = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[E] = tensor.storage();
            data[start..end]
                .iter()
                .copied()
                .reduce(|a, b| if a >= b { a } else { b })
                .expect("max: tensor must not be empty")
        }
        None => {
            let data: &[E] = tensor.storage();
            StridedIter::new(tensor.layout())
                .map(|idx| data[idx])
                .reduce(|a, b| if a >= b { a } else { b })
                .expect("max: tensor must not be empty")
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(vec![1])),
        tensor.dtype(),
    )
}

fn min_impl<E: Element + bytemuck::Pod + PartialOrd>(tensor: &FlexTensor) -> FlexTensor {
    let result: E = match tensor.layout().contiguous_offsets() {
        Some((start, end)) => {
            let data: &[E] = tensor.storage();
            data[start..end]
                .iter()
                .copied()
                .reduce(|a, b| if a <= b { a } else { b })
                .expect("min: tensor must not be empty")
        }
        None => {
            let data: &[E] = tensor.storage();
            StridedIter::new(tensor.layout())
                .map(|idx| data[idx])
                .reduce(|a, b| if a <= b { a } else { b })
                .expect("min: tensor must not be empty")
        }
    };

    let bytes = Bytes::from_elems(vec![result]);
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(vec![1])),
        tensor.dtype(),
    )
}

// ============================================================================
// Argmax / Argmin
// ============================================================================

/// Argmax along a dimension, returning indices as isize (INDEX_DTYPE).
pub fn argmax(tensor: FlexTensor, dim: usize) -> FlexTensor {
    assert_dim_fits_isize(tensor.layout().shape()[dim], dim);
    // f32 last-dim fast path: 2-pass SIMD for large rows, 1-pass scalar for small rows
    if tensor.dtype() == DType::F32 && dim == tensor.layout().shape().num_dims() - 1 {
        #[cfg(feature = "simd")]
        if tensor.layout().shape()[dim] >= EXTREMUM_SIMD_ROW_THRESHOLD {
            return extremum_indices_f32_last_simd(&tensor, dim, kernels::max_f32);
        }
        return extremum_indices_f32_last_scalar(&tensor, dim, |a, b| a > b);
    }
    match tensor.dtype() {
        DType::F32 => {
            extremum_dim_with_indices::<f32, _>(&tensor, dim, |a, b| {
                !b.is_nan() && (a.is_nan() || a > b)
            })
            .1
        }
        DType::F64 => {
            extremum_dim_with_indices::<f64, _>(&tensor, dim, |a, b| {
                !b.is_nan() && (a.is_nan() || a > b)
            })
            .1
        }
        DType::F16 => {
            extremum_dim_with_indices_half::<f16, _>(
                &tensor,
                dim,
                |a, b| !b.is_nan() && (a.is_nan() || a > b),
                f16::to_f32,
                f16::from_f32,
            )
            .1
        }
        DType::BF16 => {
            extremum_dim_with_indices_half::<bf16, _>(
                &tensor,
                dim,
                |a, b| !b.is_nan() && (a.is_nan() || a > b),
                bf16::to_f32,
                bf16::from_f32,
            )
            .1
        }
        DType::I8 => extremum_dim_with_indices::<i8, _>(&tensor, dim, |a, b| a > b).1,
        DType::I16 => extremum_dim_with_indices::<i16, _>(&tensor, dim, |a, b| a > b).1,
        DType::I32 => extremum_dim_with_indices::<i32, _>(&tensor, dim, |a, b| a > b).1,
        DType::I64 => extremum_dim_with_indices::<i64, _>(&tensor, dim, |a, b| a > b).1,
        _ => panic!("argmax: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Argmin along a dimension, returning indices as isize (INDEX_DTYPE).
pub fn argmin(tensor: FlexTensor, dim: usize) -> FlexTensor {
    assert_dim_fits_isize(tensor.layout().shape()[dim], dim);
    // f32 last-dim fast path: 2-pass SIMD for large rows, 1-pass scalar for small rows
    if tensor.dtype() == DType::F32 && dim == tensor.layout().shape().num_dims() - 1 {
        #[cfg(feature = "simd")]
        if tensor.layout().shape()[dim] >= EXTREMUM_SIMD_ROW_THRESHOLD {
            return extremum_indices_f32_last_simd(&tensor, dim, kernels::min_f32);
        }
        return extremum_indices_f32_last_scalar(&tensor, dim, |a, b| a < b);
    }
    match tensor.dtype() {
        DType::F32 => {
            extremum_dim_with_indices::<f32, _>(&tensor, dim, |a, b| {
                !b.is_nan() && (a.is_nan() || a < b)
            })
            .1
        }
        DType::F64 => {
            extremum_dim_with_indices::<f64, _>(&tensor, dim, |a, b| {
                !b.is_nan() && (a.is_nan() || a < b)
            })
            .1
        }
        DType::F16 => {
            extremum_dim_with_indices_half::<f16, _>(
                &tensor,
                dim,
                |a, b| !b.is_nan() && (a.is_nan() || a < b),
                f16::to_f32,
                f16::from_f32,
            )
            .1
        }
        DType::BF16 => {
            extremum_dim_with_indices_half::<bf16, _>(
                &tensor,
                dim,
                |a, b| !b.is_nan() && (a.is_nan() || a < b),
                bf16::to_f32,
                bf16::from_f32,
            )
            .1
        }
        DType::I8 => extremum_dim_with_indices::<i8, _>(&tensor, dim, |a, b| a < b).1,
        DType::I16 => extremum_dim_with_indices::<i16, _>(&tensor, dim, |a, b| a < b).1,
        DType::I32 => extremum_dim_with_indices::<i32, _>(&tensor, dim, |a, b| a < b).1,
        DType::I64 => extremum_dim_with_indices::<i64, _>(&tensor, dim, |a, b| a < b).1,
        _ => panic!("argmin: unsupported dtype {:?}", tensor.dtype()),
    }
}

// ============================================================================
// Dimension reduction helpers
// ============================================================================

#[derive(Clone, Copy)]
enum ReduceOp {
    Sum,
    Prod,
}

/// Optimized f32 dimension reduction with SIMD.
fn reduce_dim_f32(tensor: &FlexTensor, dim: usize, op: ReduceOp) -> FlexTensor {
    let ndims = tensor.layout().shape().num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    // Copy to contiguous only when the flattened stride assumption breaks:
    // non-contiguous tensor with 2+ outer dims or 2+ inner dims.
    let outer_dims = dim;
    let inner_dims = ndims - dim - 1;
    let needs_copy = !tensor.is_contiguous() && (outer_dims > 1 || inner_dims > 1);
    let tensor = if needs_copy {
        tensor.to_contiguous()
    } else {
        tensor.clone()
    };
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    // Empty output: any zero-sized non-reduced dim means the result has no
    // elements. Return early so the SIMD kernels and fallback loops never
    // see `outer_size == 0` or `inner_size == 0`.
    if out_size == 0 {
        return FlexTensor::new(
            Bytes::from_elems(Vec::<f32>::new()),
            Layout::contiguous(Shape::from(out_shape)),
            DType::F32,
        );
    }

    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let data: &[f32] = tensor.storage();
    let start_offset = tensor.layout().start_offset();
    let dim_stride = strides[dim];

    let (init, reduce_fn): (f32, fn(f32, f32) -> f32) = match op {
        ReduceOp::Sum => (0.0, |a, b| a + b),
        ReduceOp::Prod => (1.0, |a, b| a * b),
    };

    // Check for negative strides (from flip operations) - fall back to general case
    let has_negative_strides = strides.iter().any(|&s| s < 0);

    // Check if inner dimension is contiguous (stride = 1) and no negative strides
    let inner_contiguous = !has_negative_strides && (dim + 1 >= ndims || strides[ndims - 1] == 1);

    let result: Vec<f32> = if inner_contiguous && dim == ndims - 1 {
        // Reducing last dimension with contiguous data: use SIMD
        reduce_last_dim_f32(data, start_offset, outer_size, dim_size, strides, dim, op)
    } else if dim == 0 && inner_contiguous && matches!(op, ReduceOp::Sum) {
        // First-dim reduction with contiguous inner: use cache-friendly accumulation
        reduce_first_dim_f32(data, start_offset, dim_size, inner_size, dim_stride)
    } else if dim > 0 && dim < ndims - 1 && inner_contiguous && matches!(op, ReduceOp::Sum) {
        // Middle-dim reduction (e.g., [B, M, K] reducing dim=1): cache-friendly accumulation
        let outer_stride = strides[dim - 1];
        reduce_middle_dim_f32(
            data,
            start_offset,
            outer_size,
            dim_size,
            inner_size,
            outer_stride,
            dim_stride,
        )
    } else if dim_stride == 1 && matches!(op, ReduceOp::Sum) && outer_size == 1 {
        // Reduction dimension is contiguous, no outer batch (e.g., transposed 2D reducing dim=0)
        // Storage is [inner_size rows of dim_size elements each] - use sum_rows_f32
        #[cfg(feature = "simd")]
        {
            let mut result = vec![0.0f32; inner_size];
            kernels::sum_rows_f32(
                &data[start_offset..],
                &mut result,
                inner_size, // number of rows (output positions)
                dim_size,   // elements per row (to sum)
            );
            result
        }
        #[cfg(not(feature = "simd"))]
        {
            let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };
            let mut result = Vec::with_capacity(out_size);
            for inner in 0..inner_size {
                let base = (start_offset as isize + inner as isize * inner_stride) as usize;
                let slice = &data[base..base + dim_size];
                result.push(slice.iter().copied().sum());
            }
            result
        }
    } else if dim_stride == 1 && matches!(op, ReduceOp::Sum) {
        // Reduction dimension is contiguous but with outer batches
        let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
        let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

        let mut result = Vec::with_capacity(out_size);
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base = (start_offset as isize
                    + outer as isize * outer_stride
                    + inner as isize * inner_stride) as usize;
                let slice = &data[base..base + dim_size];
                #[cfg(feature = "simd")]
                let acc = kernels::sum_f32(slice);
                #[cfg(not(feature = "simd"))]
                let acc = slice.iter().copied().sum();
                result.push(acc);
            }
        }
        result
    } else if tensor.is_contiguous() {
        // Contiguous: use flat index arithmetic (safe for any ndims).
        // outer_size and inner_size are guaranteed positive by the out_size == 0
        // early return above.
        let mut result = Vec::with_capacity(out_size);
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut acc = init;
                for d in 0..dim_size {
                    let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                    acc = reduce_fn(acc, data[idx]);
                }
                result.push(acc);
            }
        }
        result
    } else {
        // Non-contiguous with at most 1 outer + 1 inner dim (e.g., flipped 2D)
        let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
        let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

        let mut result = Vec::with_capacity(out_size);
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base = start_offset as isize
                    + outer as isize * outer_stride
                    + inner as isize * inner_stride;
                let mut acc = init;
                for d in 0..dim_size {
                    let idx = (base + d as isize * dim_stride) as usize;
                    acc = reduce_fn(acc, data[idx]);
                }
                result.push(acc);
            }
        }
        result
    };

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        DType::F32,
    )
}

/// Reduce middle dimension (e.g., [B, M, K] reducing dim=1) with cache-friendly iteration.
/// For each batch, iterate over rows (dim to reduce) sequentially and accumulate into columns.
#[inline]
fn reduce_middle_dim_f32(
    data: &[f32],
    start_offset: usize,
    outer_size: usize, // batch size
    dim_size: usize,   // rows to sum
    inner_size: usize, // columns (output per batch)
    outer_stride: isize,
    dim_stride: isize,
) -> Vec<f32> {
    let out_size = outer_size * inner_size;

    #[cfg(feature = "simd")]
    {
        // Use aligned allocation for optimal SIMD scatter-add
        let mut result = aligned::alloc_aligned_zeroed::<f32>(out_size);
        kernels::scatter_add_batched(
            &data[start_offset..],
            &mut result,
            outer_size,
            dim_size,
            inner_size,
            outer_stride as usize,
            dim_stride as usize,
        );
        aligned::to_vec(result)
    }

    #[cfg(not(feature = "simd"))]
    {
        let mut result = vec![0.0f32; out_size];
        let start = start_offset as isize;
        for batch in 0..outer_size {
            let batch_start = (start + batch as isize * outer_stride) as usize;
            let out_batch_start = batch * inner_size;

            for row in 0..dim_size {
                let row_start = (batch_start as isize + row as isize * dim_stride) as usize;
                for c in 0..inner_size {
                    result[out_batch_start + c] += data[row_start + c];
                }
            }
        }
        result
    }
}

/// Reduce first dimension with cache-friendly row iteration.
/// Instead of iterating per-output (col) and gathering from rows (cache-unfriendly),
/// iterate over rows (sequential access) and scatter-accumulate into outputs.
#[inline]
fn reduce_first_dim_f32(
    data: &[f32],
    start_offset: usize,
    dim_size: usize,   // number of rows to sum
    inner_size: usize, // number of columns (output positions)
    dim_stride: isize, // stride between rows
) -> Vec<f32> {
    #[cfg(feature = "simd")]
    {
        // Use aligned allocation for optimal SIMD scatter-add
        let mut result = aligned::alloc_aligned_zeroed::<f32>(inner_size);
        kernels::scatter_add_f32(
            &data[start_offset..],
            &mut result,
            dim_size,
            inner_size,
            dim_stride as usize,
        );
        aligned::to_vec(result)
    }

    #[cfg(not(feature = "simd"))]
    {
        let mut result = vec![0.0f32; inner_size];
        let start = start_offset as isize;
        for row in 0..dim_size {
            let row_start = (start + row as isize * dim_stride) as usize;
            for c in 0..inner_size {
                result[c] += data[row_start + c];
            }
        }
        result
    }
}

/// Reduce last dimension with SIMD.
///
/// For contiguous Sum: batches all rows in a single kernel call using
/// 4-accumulator SIMD to hide add latency.
#[inline]
fn reduce_last_dim_f32(
    data: &[f32],
    start_offset: usize,
    outer_size: usize,
    dim_size: usize,
    strides: &[isize],
    dim: usize,
    op: ReduceOp,
) -> Vec<f32> {
    let outer_stride: isize = if dim > 0 {
        strides[dim - 1]
    } else {
        dim_size as isize
    };

    // `outer_size > 0` is guaranteed by the out_size == 0 early return in
    // `reduce_dim_f32`.
    let rows = outer_size;

    // Contiguous Sum: batch all rows in one kernel call to avoid per-row overhead.
    #[cfg(feature = "simd")]
    if matches!(op, ReduceOp::Sum) && outer_stride == dim_size as isize {
        let mut result = vec![0.0f32; rows];
        kernels::sum_rows_f32(&data[start_offset..], &mut result, rows, dim_size);
        return result;
    }

    // Fallback: non-contiguous strides or Prod.
    let mut result = Vec::with_capacity(rows);
    for outer in 0..rows {
        let row_start = (start_offset as isize + outer as isize * outer_stride) as usize;
        let row = &data[row_start..row_start + dim_size];

        let val = match op {
            ReduceOp::Sum => {
                #[cfg(feature = "simd")]
                {
                    kernels::sum_f32(row)
                }
                #[cfg(not(feature = "simd"))]
                {
                    row.iter().copied().sum()
                }
            }
            ReduceOp::Prod => row.iter().copied().product(),
        };
        result.push(val);
    }
    result
}

/// Generic dimension reduction implementation.
fn reduce_dim_impl<E, F>(tensor: &FlexTensor, dim: usize, init: E, reduce_fn: F) -> FlexTensor
where
    E: Element + bytemuck::Pod + Copy,
    F: Fn(E, E) -> E,
{
    let ndims = tensor.layout().shape().num_dims();
    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    // Copy to contiguous only when the flattened stride assumption breaks:
    // non-contiguous tensor with 2+ outer dims or 2+ inner dims.
    let outer_dims = dim;
    let inner_dims = ndims - dim - 1;
    let needs_copy = !tensor.is_contiguous() && (outer_dims > 1 || inner_dims > 1);
    let tensor = if needs_copy {
        tensor.to_contiguous()
    } else {
        tensor.clone()
    };
    let shape = tensor.layout().shape();
    let strides = tensor.layout().strides();

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    // Empty output: any zero-sized non-reduced dim means the result has no
    // elements. Returning early keeps the loops below from producing phantom
    // outputs when `outer_size == 0` or `inner_size == 0`.
    if out_size == 0 {
        return FlexTensor::new(
            Bytes::from_elems(Vec::<E>::new()),
            Layout::contiguous(Shape::from(out_shape)),
            tensor.dtype(),
        );
    }

    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut result: Vec<E> = Vec::with_capacity(out_size);

    if tensor.is_contiguous() {
        // Contiguous: use flat index arithmetic (safe for any ndims).
        // outer_size and inner_size are positive by the early return above.
        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut acc = init;
                for d in 0..dim_size {
                    let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                    acc = reduce_fn(acc, data[idx]);
                }
                result.push(acc);
            }
        }
    } else {
        // Non-contiguous with at most 1 outer + 1 inner dim
        let dim_stride = strides[dim];
        let outer_stride: isize = if dim > 0 { strides[dim - 1] } else { 0 };
        let inner_stride: isize = if dim + 1 < ndims { strides[dim + 1] } else { 1 };

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base = start_offset as isize
                    + outer as isize * outer_stride
                    + inner as isize * inner_stride;
                let mut acc = init;
                for d in 0..dim_size {
                    let idx = (base + d as isize * dim_stride) as usize;
                    acc = reduce_fn(acc, data[idx]);
                }
                result.push(acc);
            }
        }
    }

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        tensor.dtype(),
    )
}

/// Widening dimension reduction for small integer types: accumulate in i64 to avoid overflow.
fn reduce_dim_widening<E, F>(tensor: &FlexTensor, dim: usize, init: i64, reduce_fn: F) -> FlexTensor
where
    E: Element + bytemuck::Pod,
    i64: From<E>,
    F: Fn(i64, i64) -> i64,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    // Empty output: skip the loop entirely for zero-sized non-reduced dims.
    if out_size == 0 {
        return FlexTensor::new(
            Bytes::from_elems(Vec::<E>::new()),
            Layout::contiguous(Shape::from(out_shape)),
            tensor.dtype(),
        );
    }

    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut result: Vec<E> = Vec::with_capacity(out_size);

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = init;
            for d in 0..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                acc = reduce_fn(acc, i64::from(data[idx]));
            }
            // Truncate back to target type (wrapping, matches PyTorch)
            let val: E = truncate_i64_to_pod(acc);
            result.push(val);
        }
    }

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        tensor.dtype(),
    )
}

/// Half-precision dimension reduction with f32 accumulation.
///
/// Works for both f16 and bf16 via the `to_f32`/`from_f32` closures.
fn reduce_dim_half<E, F>(
    tensor: &FlexTensor,
    dim: usize,
    init: f32,
    reduce_fn: F,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> FlexTensor
where
    E: Element + bytemuck::Pod,
    F: Fn(f32, f32) -> f32,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let out_size: usize = out_shape.iter().product();

    // Empty output: skip the loop entirely for zero-sized non-reduced dims.
    if out_size == 0 {
        return FlexTensor::new(
            Bytes::from_elems(Vec::<E>::new()),
            Layout::contiguous(Shape::from(out_shape)),
            E::dtype(),
        );
    }

    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let mut result: Vec<E> = Vec::with_capacity(out_size);

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = init;
            for d in 0..dim_size {
                let idx = start_offset + outer * dim_size * inner_size + d * inner_size + inner;
                acc = reduce_fn(acc, to_f32(data[idx]));
            }
            result.push(from_f32(acc));
        }
    }

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        E::dtype(),
    )
}

/// Sum along `dim` for an already-contiguous row-major f32 slice, producing
/// one output per (outer, inner) position.
///
/// The caller owns the contiguity guarantee: `data.len()` must equal
/// `outer_size * dim_size * inner_size` in logical row-major order. Dispatches
/// to the same SIMD kernels `reduce_dim_f32` uses, but without the stride
/// bookkeeping.
fn sum_dim_contiguous_f32(
    data: &[f32],
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
) -> Vec<f32> {
    // Empty output: if any non-reduced dim has size 0, the result is empty.
    // Returning early avoids indexing past an empty `data` slice in the SIMD
    // kernels and keeps the output length in sync with the caller's out_shape.
    if outer_size == 0 || inner_size == 0 {
        return Vec::new();
    }

    // Last-dim: each output is the sum of a contiguous run of dim_size elements.
    if inner_size == 1 {
        let rows = outer_size;
        #[cfg(feature = "simd")]
        {
            let mut result = vec![0.0f32; rows];
            kernels::sum_rows_f32(data, &mut result, rows, dim_size);
            return result;
        }
        #[cfg(not(feature = "simd"))]
        {
            return (0..rows)
                .map(|i| data[i * dim_size..(i + 1) * dim_size].iter().sum())
                .collect();
        }
    }

    // First-dim (or equivalent: any collapsed-outer case): scatter-add dim_size
    // rows of inner_size cols into a single inner_size accumulator.
    if outer_size == 1 {
        #[cfg(feature = "simd")]
        {
            let mut result = aligned::alloc_aligned_zeroed::<f32>(inner_size);
            kernels::scatter_add_f32(data, &mut result, dim_size, inner_size, inner_size);
            return aligned::to_vec(result);
        }
        #[cfg(not(feature = "simd"))]
        {
            let mut result = vec![0.0f32; inner_size];
            for row in 0..dim_size {
                let row_start = row * inner_size;
                for c in 0..inner_size {
                    result[c] += data[row_start + c];
                }
            }
            return result;
        }
    }

    // Middle-dim: batched scatter-add. For each outer batch, sum dim_size rows
    // of inner_size cols into a per-batch accumulator.
    let out_size = outer_size * inner_size;
    #[cfg(feature = "simd")]
    {
        let mut result = aligned::alloc_aligned_zeroed::<f32>(out_size);
        kernels::scatter_add_batched(
            data,
            &mut result,
            outer_size,
            dim_size,
            inner_size,
            dim_size * inner_size,
            inner_size,
        );
        aligned::to_vec(result)
    }
    #[cfg(not(feature = "simd"))]
    {
        let mut result = vec![0.0f32; out_size];
        for outer in 0..outer_size {
            let out_base = outer * inner_size;
            for d in 0..dim_size {
                let in_base = outer * dim_size * inner_size + d * inner_size;
                for c in 0..inner_size {
                    result[out_base + c] += data[in_base + c];
                }
            }
        }
        result
    }
}

/// Mean along a dimension for half-precision types, fusing sum and divide in f32.
///
/// A naive `sum_dim` + `scalar_div` implementation can overflow to +inf when the
/// intermediate sum exceeds `f16::MAX` (65504), even if the final mean fits. This
/// function keeps the entire reduction and division in f32 and only narrows to
/// f16/bf16 on store via `E::from_elem`.
fn mean_dim_half<E>(tensor: &FlexTensor, dim: usize) -> FlexTensor
where
    E: Element + bytemuck::Pod,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    let dim_size = shape[dim];
    assert!(
        dim_size > 0,
        "mean_dim: cannot take mean of empty dimension"
    );
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;

    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    let data = float_storage_as_f32(&tensor);
    let divisor = dim_size as f32;

    let sums = sum_dim_contiguous_f32(&data, outer_size, dim_size, inner_size);
    let result: Vec<E> = sums
        .into_iter()
        .map(|s| E::from_elem(s / divisor))
        .collect();

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(
        bytes,
        Layout::contiguous(Shape::from(out_shape)),
        E::dtype(),
    )
}

/// Scalar mean for half-precision types, fusing sum and divide in f32.
/// Avoids f16 overflow when the total sum exceeds `f16::MAX`. Empty input
/// produces NaN to match the f32/f64 path in `mean()`.
fn mean_scalar_half<E>(tensor: &FlexTensor) -> FlexTensor
where
    E: Element + bytemuck::Pod,
{
    let tensor = tensor.to_contiguous();
    let n = tensor.layout().num_elements();
    let data = float_storage_as_f32(&tensor);
    // Route through `sum_f32_contiguous` to pick up SIMD + rayon for the
    // f32 reduction. The half-precision narrowing happens after the divide.
    let acc = sum_f32_contiguous(&data);

    let mean = acc / (n as f32);
    let bytes = Bytes::from_elems(vec![E::from_elem(mean)]);
    FlexTensor::new(bytes, Layout::contiguous(Shape::from(vec![1])), E::dtype())
}

// ============================================================================
// Mean (all elements)
// ============================================================================

/// Mean of all elements, returning a scalar tensor.
pub fn mean(tensor: FlexTensor) -> FlexTensor {
    let dtype = tensor.dtype();

    // Half-precision types fuse sum+divide in f32 to avoid overflow when the
    // total sum exceeds f16::MAX.
    match dtype {
        DType::F16 => return mean_scalar_half::<f16>(&tensor),
        DType::BF16 => return mean_scalar_half::<bf16>(&tensor),
        _ => {}
    }

    let n = tensor.layout().num_elements();
    let sum_result = sum(tensor);
    match dtype {
        DType::F32 => scalar_div::<f32>(sum_result, n as f32),
        DType::F64 => scalar_div::<f64>(sum_result, n as f64),
        _ => panic!("mean: unsupported dtype {:?}", dtype),
    }
}

// ============================================================================
// Max/Min along dimension (value + optional indices in a single pass)
// ============================================================================

/// Max along a dimension, returning only values.
pub fn max_dim(tensor: FlexTensor, dim: usize) -> FlexTensor {
    assert!(
        tensor.layout().shape()[dim] > 0,
        "max_dim: dimension {dim} has size 0"
    );
    if tensor.dtype() == DType::F32 && dim == tensor.layout().shape().num_dims() - 1 {
        #[cfg(feature = "simd")]
        if tensor.layout().shape()[dim] >= EXTREMUM_SIMD_ROW_THRESHOLD {
            return extremum_dim_f32_last_simd(&tensor, dim, kernels::max_f32);
        }
        return extremum_f32_last_scalar(&tensor, dim, |a, b| a > b);
    }
    match tensor.dtype() {
        DType::F32 => {
            extremum_dim::<f32, _>(&tensor, dim, |a, b| !b.is_nan() && (a.is_nan() || a > b))
        }
        DType::F64 => {
            extremum_dim::<f64, _>(&tensor, dim, |a, b| !b.is_nan() && (a.is_nan() || a > b))
        }
        DType::F16 => extremum_dim_half::<f16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a > b),
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => extremum_dim_half::<bf16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a > b),
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I64 => extremum_dim::<i64, _>(&tensor, dim, |a, b| a > b),
        DType::I32 => extremum_dim::<i32, _>(&tensor, dim, |a, b| a > b),
        DType::I16 => extremum_dim::<i16, _>(&tensor, dim, |a, b| a > b),
        DType::I8 => extremum_dim::<i8, _>(&tensor, dim, |a, b| a > b),
        DType::U64 => extremum_dim::<u64, _>(&tensor, dim, |a, b| a > b),
        DType::U32 => extremum_dim::<u32, _>(&tensor, dim, |a, b| a > b),
        DType::U16 => extremum_dim::<u16, _>(&tensor, dim, |a, b| a > b),
        DType::U8 => extremum_dim::<u8, _>(&tensor, dim, |a, b| a > b),
        _ => panic!("max_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Min along a dimension, returning only values.
pub fn min_dim(tensor: FlexTensor, dim: usize) -> FlexTensor {
    assert!(
        tensor.layout().shape()[dim] > 0,
        "min_dim: dimension {dim} has size 0"
    );
    if tensor.dtype() == DType::F32 && dim == tensor.layout().shape().num_dims() - 1 {
        #[cfg(feature = "simd")]
        if tensor.layout().shape()[dim] >= EXTREMUM_SIMD_ROW_THRESHOLD {
            return extremum_dim_f32_last_simd(&tensor, dim, kernels::min_f32);
        }
        return extremum_f32_last_scalar(&tensor, dim, |a, b| a < b);
    }
    match tensor.dtype() {
        DType::F32 => {
            extremum_dim::<f32, _>(&tensor, dim, |a, b| !b.is_nan() && (a.is_nan() || a < b))
        }
        DType::F64 => {
            extremum_dim::<f64, _>(&tensor, dim, |a, b| !b.is_nan() && (a.is_nan() || a < b))
        }
        DType::F16 => extremum_dim_half::<f16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a < b),
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => extremum_dim_half::<bf16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a < b),
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I64 => extremum_dim::<i64, _>(&tensor, dim, |a, b| a < b),
        DType::I32 => extremum_dim::<i32, _>(&tensor, dim, |a, b| a < b),
        DType::I16 => extremum_dim::<i16, _>(&tensor, dim, |a, b| a < b),
        DType::I8 => extremum_dim::<i8, _>(&tensor, dim, |a, b| a < b),
        DType::U64 => extremum_dim::<u64, _>(&tensor, dim, |a, b| a < b),
        DType::U32 => extremum_dim::<u32, _>(&tensor, dim, |a, b| a < b),
        DType::U16 => extremum_dim::<u16, _>(&tensor, dim, |a, b| a < b),
        DType::U8 => extremum_dim::<u8, _>(&tensor, dim, |a, b| a < b),
        _ => panic!("min_dim: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Max along a dimension with indices, returning (values, indices) in a single pass.
pub fn max_dim_with_indices(tensor: FlexTensor, dim: usize) -> (FlexTensor, FlexTensor) {
    let dim_len = tensor.layout().shape()[dim];
    assert!(
        dim_len > 0,
        "max_dim_with_indices: dimension {dim} has size 0"
    );
    assert_dim_fits_isize(dim_len, dim);
    if tensor.dtype() == DType::F32 && dim == tensor.layout().shape().num_dims() - 1 {
        #[cfg(feature = "simd")]
        if tensor.layout().shape()[dim] >= EXTREMUM_SIMD_ROW_THRESHOLD {
            return extremum_dim_with_indices_f32_last_simd(&tensor, dim, kernels::max_f32);
        }
        return extremum_with_indices_f32_last_scalar(&tensor, dim, |a, b| a > b);
    }
    match tensor.dtype() {
        DType::F32 => extremum_dim_with_indices::<f32, _>(&tensor, dim, |a, b| {
            !b.is_nan() && (a.is_nan() || a > b)
        }),
        DType::F64 => extremum_dim_with_indices::<f64, _>(&tensor, dim, |a, b| {
            !b.is_nan() && (a.is_nan() || a > b)
        }),
        DType::F16 => extremum_dim_with_indices_half::<f16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a > b),
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => extremum_dim_with_indices_half::<bf16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a > b),
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I64 => extremum_dim_with_indices::<i64, _>(&tensor, dim, |a, b| a > b),
        DType::I32 => extremum_dim_with_indices::<i32, _>(&tensor, dim, |a, b| a > b),
        DType::I16 => extremum_dim_with_indices::<i16, _>(&tensor, dim, |a, b| a > b),
        DType::I8 => extremum_dim_with_indices::<i8, _>(&tensor, dim, |a, b| a > b),
        DType::U64 => extremum_dim_with_indices::<u64, _>(&tensor, dim, |a, b| a > b),
        DType::U32 => extremum_dim_with_indices::<u32, _>(&tensor, dim, |a, b| a > b),
        DType::U16 => extremum_dim_with_indices::<u16, _>(&tensor, dim, |a, b| a > b),
        DType::U8 => extremum_dim_with_indices::<u8, _>(&tensor, dim, |a, b| a > b),
        _ => panic!(
            "max_dim_with_indices: unsupported dtype {:?}",
            tensor.dtype()
        ),
    }
}

/// Min along a dimension with indices, returning (values, indices) in a single pass.
pub fn min_dim_with_indices(tensor: FlexTensor, dim: usize) -> (FlexTensor, FlexTensor) {
    let dim_len = tensor.layout().shape()[dim];
    assert!(
        dim_len > 0,
        "min_dim_with_indices: dimension {dim} has size 0"
    );
    assert_dim_fits_isize(dim_len, dim);
    if tensor.dtype() == DType::F32 && dim == tensor.layout().shape().num_dims() - 1 {
        #[cfg(feature = "simd")]
        if tensor.layout().shape()[dim] >= EXTREMUM_SIMD_ROW_THRESHOLD {
            return extremum_dim_with_indices_f32_last_simd(&tensor, dim, kernels::min_f32);
        }
        return extremum_with_indices_f32_last_scalar(&tensor, dim, |a, b| a < b);
    }
    match tensor.dtype() {
        DType::F32 => extremum_dim_with_indices::<f32, _>(&tensor, dim, |a, b| {
            !b.is_nan() && (a.is_nan() || a < b)
        }),
        DType::F64 => extremum_dim_with_indices::<f64, _>(&tensor, dim, |a, b| {
            !b.is_nan() && (a.is_nan() || a < b)
        }),
        DType::F16 => extremum_dim_with_indices_half::<f16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a < b),
            f16::to_f32,
            f16::from_f32,
        ),
        DType::BF16 => extremum_dim_with_indices_half::<bf16, _>(
            &tensor,
            dim,
            |a, b| !b.is_nan() && (a.is_nan() || a < b),
            bf16::to_f32,
            bf16::from_f32,
        ),
        DType::I64 => extremum_dim_with_indices::<i64, _>(&tensor, dim, |a, b| a < b),
        DType::I32 => extremum_dim_with_indices::<i32, _>(&tensor, dim, |a, b| a < b),
        DType::I16 => extremum_dim_with_indices::<i16, _>(&tensor, dim, |a, b| a < b),
        DType::I8 => extremum_dim_with_indices::<i8, _>(&tensor, dim, |a, b| a < b),
        DType::U64 => extremum_dim_with_indices::<u64, _>(&tensor, dim, |a, b| a < b),
        DType::U32 => extremum_dim_with_indices::<u32, _>(&tensor, dim, |a, b| a < b),
        DType::U16 => extremum_dim_with_indices::<u16, _>(&tensor, dim, |a, b| a < b),
        DType::U8 => extremum_dim_with_indices::<u8, _>(&tensor, dim, |a, b| a < b),
        _ => panic!(
            "min_dim_with_indices: unsupported dtype {:?}",
            tensor.dtype()
        ),
    }
}

// ============================================================================
// Extremum helpers (SIMD fast paths + generic scalar, parallelized with rayon)
// ============================================================================

// Lower threshold than the global PARALLEL_THRESHOLD (256K) because the per-element
// work (a single comparison + conditional store) is cheap enough that rayon overhead
// is amortized at smaller sizes. 32K elements * ~1.5ns/elem = ~48µs of serial work,
// enough to justify thread-pool dispatch.
#[cfg(feature = "rayon")]
const EXTREMUM_PARALLEL_THRESHOLD: usize = 32 * 1024;

/// Minimum row length for the 2-pass SIMD extremum path (reduce + scan).
/// Below this, single-pass scalar is faster since it reads each element once.
#[cfg(feature = "simd")]
const EXTREMUM_SIMD_ROW_THRESHOLD: usize = 512;

/// Scalar single-pass f32 last-dim extremum (values only).
/// Avoids the generic closure path by operating directly on contiguous f32 rows.
fn extremum_f32_last_scalar<F>(tensor: &FlexTensor, dim: usize, is_better: F) -> FlexTensor
where
    F: Fn(f32, f32) -> bool + Send + Sync,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let dim_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let data: &[f32] = tensor.storage();
    let start = tensor.layout().start_offset();

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;

    let reduce_row = |outer: usize| -> f32 {
        let row_start = start + outer * dim_size;
        let row = &data[row_start..row_start + dim_size];
        let mut best = row[0];
        for &v in &row[1..] {
            if v.is_nan() {
                return f32::NAN;
            }
            if is_better(v, best) {
                best = v;
            }
        }
        // Check first element for NaN (skipped in loop)
        if best.is_nan() {
            return f32::NAN;
        }
        best
    };

    #[cfg(feature = "rayon")]
    let values: Vec<f32> = if outer_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
        (0..outer_size).into_par_iter().map(&reduce_row).collect()
    } else {
        (0..outer_size).map(reduce_row).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let values: Vec<f32> = (0..outer_size).map(reduce_row).collect();

    FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        DType::F32,
    )
}

/// Scalar single-pass f32 last-dim argmax/argmin (indices only).
/// Uses direct pointer access and simple comparisons instead of the generic closure path.
fn extremum_indices_f32_last_scalar<F>(tensor: &FlexTensor, dim: usize, is_better: F) -> FlexTensor
where
    F: Fn(f32, f32) -> bool + Send + Sync,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let dim_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let data: &[f32] = tensor.storage();
    let start = tensor.layout().start_offset();

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;

    let find_row = |outer: usize| -> isize {
        let row_start = start + outer * dim_size;
        let row = &data[row_start..row_start + dim_size];
        let mut best = row[0];
        let mut best_idx: isize = 0;
        for (i, &v) in row[1..].iter().enumerate() {
            if v.is_nan() {
                return (i + 1) as isize;
            }
            if is_better(v, best) {
                best = v;
                best_idx = (i + 1) as isize;
            }
        }
        // Check first element for NaN (skipped in loop)
        if row[0].is_nan() {
            return 0;
        }
        best_idx
    };

    #[cfg(feature = "rayon")]
    let indices: Vec<isize> = if outer_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
        (0..outer_size).into_par_iter().map(find_row).collect()
    } else {
        (0..outer_size).map(find_row).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let indices: Vec<isize> = (0..outer_size).map(find_row).collect();

    FlexTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        INDEX_DTYPE,
    )
}

/// Scalar single-pass f32 last-dim extremum with indices (values + indices).
fn extremum_with_indices_f32_last_scalar<F>(
    tensor: &FlexTensor,
    dim: usize,
    is_better: F,
) -> (FlexTensor, FlexTensor)
where
    F: Fn(f32, f32) -> bool + Send + Sync,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let dim_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let data: &[f32] = tensor.storage();
    let start = tensor.layout().start_offset();

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;

    let find_row = |outer: usize| -> (f32, isize) {
        let row_start = start + outer * dim_size;
        let row = &data[row_start..row_start + dim_size];
        let mut best = row[0];
        let mut best_idx: isize = 0;
        for (i, &v) in row[1..].iter().enumerate() {
            if v.is_nan() {
                return (f32::NAN, (i + 1) as isize);
            }
            if is_better(v, best) {
                best = v;
                best_idx = (i + 1) as isize;
            }
        }
        if row[0].is_nan() {
            return (f32::NAN, 0);
        }
        (best, best_idx)
    };

    #[cfg(feature = "rayon")]
    let (values, indices): (Vec<f32>, Vec<isize>) =
        if outer_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
            (0..outer_size).into_par_iter().map(find_row).unzip()
        } else {
            (0..outer_size).map(find_row).unzip()
        };

    #[cfg(not(feature = "rayon"))]
    let (values, indices): (Vec<f32>, Vec<isize>) = (0..outer_size).map(find_row).unzip();

    let val_tensor = FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape.clone())),
        DType::F32,
    );
    let idx_tensor = FlexTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        INDEX_DTYPE,
    );
    (val_tensor, idx_tensor)
}

/// Uses macerator SIMD reduction per contiguous row, with NaN propagation.
#[cfg(feature = "simd")]
fn extremum_dim_f32_last_simd(
    tensor: &FlexTensor,
    dim: usize,
    simd_reduce: fn(&[f32]) -> f32,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let dim_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let data: &[f32] = tensor.storage();
    let start = tensor.layout().start_offset();

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;

    let reduce_row = |outer: usize| -> f32 {
        let row_start = start + outer * dim_size;
        let row = &data[row_start..row_start + dim_size];
        let ext = simd_reduce(row);
        // SIMD max/min may silently drop NaN (architecture-dependent).
        // If the result is already NaN, we're done. Otherwise, scan to
        // check for any NaN the SIMD op missed.
        if ext.is_nan() {
            return f32::NAN;
        }
        for &v in row {
            if v.is_nan() {
                return f32::NAN;
            }
        }
        ext
    };

    #[cfg(feature = "rayon")]
    let values: Vec<f32> = if outer_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
        (0..outer_size).into_par_iter().map(reduce_row).collect()
    } else {
        (0..outer_size).map(reduce_row).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let values: Vec<f32> = (0..outer_size).map(reduce_row).collect();

    FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        DType::F32,
    )
}

/// SIMD fast path for f32 last-dim extremum with indices.
/// Per row: SIMD reduction finds the extremum value, then a linear scan
/// locates the first NaN or first matching index.
#[cfg(feature = "simd")]
fn extremum_dim_with_indices_f32_last_simd(
    tensor: &FlexTensor,
    dim: usize,
    simd_reduce: fn(&[f32]) -> f32,
) -> (FlexTensor, FlexTensor) {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let dim_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let data: &[f32] = tensor.storage();
    let start = tensor.layout().start_offset();

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;

    let find_row = |outer: usize| -> (f32, isize) {
        let row_start = start + outer * dim_size;
        let row = &data[row_start..row_start + dim_size];
        let ext = simd_reduce(row);
        // Single scan: return first NaN (with NaN value) or first match of ext.
        for (i, &v) in row.iter().enumerate() {
            if v.is_nan() {
                return (f32::NAN, i as isize);
            }
            if v == ext {
                return (ext, i as isize);
            }
        }
        (ext, 0)
    };

    #[cfg(feature = "rayon")]
    let (values, indices): (Vec<f32>, Vec<isize>) =
        if outer_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
            (0..outer_size).into_par_iter().map(find_row).unzip()
        } else {
            (0..outer_size).map(find_row).unzip()
        };

    #[cfg(not(feature = "rayon"))]
    let (values, indices): (Vec<f32>, Vec<isize>) = (0..outer_size).map(find_row).unzip();

    let val_tensor = FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape.clone())),
        DType::F32,
    );
    let idx_tensor = FlexTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        INDEX_DTYPE,
    );
    (val_tensor, idx_tensor)
}

/// SIMD fast path for f32 last-dim argmax/argmin (indices only, no values allocation).
#[cfg(feature = "simd")]
fn extremum_indices_f32_last_simd(
    tensor: &FlexTensor,
    dim: usize,
    simd_reduce: fn(&[f32]) -> f32,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let dim_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let data: &[f32] = tensor.storage();
    let start = tensor.layout().start_offset();

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;

    let find_row = |outer: usize| -> isize {
        let row_start = start + outer * dim_size;
        let row = &data[row_start..row_start + dim_size];
        let ext = simd_reduce(row);
        for (i, &v) in row.iter().enumerate() {
            if v.is_nan() || v == ext {
                return i as isize;
            }
        }
        0
    };

    #[cfg(feature = "rayon")]
    let indices: Vec<isize> = if outer_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
        (0..outer_size).into_par_iter().map(find_row).collect()
    } else {
        (0..outer_size).map(find_row).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let indices: Vec<isize> = (0..outer_size).map(find_row).collect();

    FlexTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        INDEX_DTYPE,
    )
}

/// Find extremum value along a dimension. `is_better(new, current) -> bool`.
fn extremum_dim<E, F>(tensor: &FlexTensor, dim: usize, is_better: F) -> FlexTensor
where
    E: Element + bytemuck::Pod + Send + Sync,
    F: Fn(E, E) -> bool + Send + Sync,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let outer_size: usize = shape[..dim].iter().product::<usize>();
    let inner_size: usize = shape[dim + 1..].iter().product::<usize>();
    let out_size = outer_size * inner_size;
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let find = |flat_idx: usize| -> E {
        let outer = flat_idx / inner_size;
        let inner = flat_idx % inner_size;
        let base = start_offset + outer * dim_size * inner_size + inner;
        let mut best = data[base];
        for d in 1..dim_size {
            let val = data[base + d * inner_size];
            if is_better(val, best) {
                best = val;
            }
        }
        best
    };

    #[cfg(feature = "rayon")]
    let values: Vec<E> = if out_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
        (0..out_size).into_par_iter().map(&find).collect()
    } else {
        (0..out_size).map(find).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let values: Vec<E> = (0..out_size).map(find).collect();

    FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        E::dtype(),
    )
}

/// Find extremum value and its index along a dimension. `is_better(new, current) -> bool`.
fn extremum_dim_with_indices<E, F>(
    tensor: &FlexTensor,
    dim: usize,
    is_better: F,
) -> (FlexTensor, FlexTensor)
where
    E: Element + bytemuck::Pod + Send + Sync,
    F: Fn(E, E) -> bool + Send + Sync,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let outer_size: usize = shape[..dim].iter().product::<usize>();
    let inner_size: usize = shape[dim + 1..].iter().product::<usize>();
    let out_size = outer_size * inner_size;
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let find = |flat_idx: usize| -> (E, isize) {
        let outer = flat_idx / inner_size;
        let inner = flat_idx % inner_size;
        let base = start_offset + outer * dim_size * inner_size + inner;
        let mut best = data[base];
        let mut best_idx: isize = 0;
        for d in 1..dim_size {
            let val = data[base + d * inner_size];
            if is_better(val, best) {
                best = val;
                best_idx = d as isize;
            }
        }
        (best, best_idx)
    };

    #[cfg(feature = "rayon")]
    let (values, indices): (Vec<E>, Vec<isize>) =
        if out_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
            (0..out_size).into_par_iter().map(&find).unzip()
        } else {
            (0..out_size).map(find).unzip()
        };

    #[cfg(not(feature = "rayon"))]
    let (values, indices): (Vec<E>, Vec<isize>) = (0..out_size).map(find).unzip();

    let val_tensor = FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape.clone())),
        E::dtype(),
    );
    let idx_tensor = FlexTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        INDEX_DTYPE,
    );
    (val_tensor, idx_tensor)
}

/// Find extremum value along a dimension for half-precision types (compared via f32).
fn extremum_dim_half<E, F>(
    tensor: &FlexTensor,
    dim: usize,
    is_better: F,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> FlexTensor
where
    E: Element + bytemuck::Pod + Send + Sync,
    F: Fn(f32, f32) -> bool + Send + Sync,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let outer_size: usize = shape[..dim].iter().product::<usize>();
    let inner_size: usize = shape[dim + 1..].iter().product::<usize>();
    let out_size = outer_size * inner_size;
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let find = |flat_idx: usize| -> E {
        let outer = flat_idx / inner_size;
        let inner = flat_idx % inner_size;
        let base = start_offset + outer * dim_size * inner_size + inner;
        let mut best = to_f32(data[base]);
        for d in 1..dim_size {
            let val = to_f32(data[base + d * inner_size]);
            if is_better(val, best) {
                best = val;
            }
        }
        from_f32(best)
    };

    #[cfg(feature = "rayon")]
    let values: Vec<E> = if out_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
        (0..out_size).into_par_iter().map(&find).collect()
    } else {
        (0..out_size).map(find).collect()
    };

    #[cfg(not(feature = "rayon"))]
    let values: Vec<E> = (0..out_size).map(find).collect();

    FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape)),
        E::dtype(),
    )
}

/// Find extremum value and index along a dimension for half-precision types (compared via f32).
fn extremum_dim_with_indices_half<E, F>(
    tensor: &FlexTensor,
    dim: usize,
    is_better: F,
    to_f32: fn(E) -> f32,
    from_f32: fn(f32) -> E,
) -> (FlexTensor, FlexTensor)
where
    E: Element + bytemuck::Pod + Send + Sync,
    F: Fn(f32, f32) -> bool + Send + Sync,
{
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape();
    let ndims = shape.num_dims();
    assert!(dim < ndims);

    let dim_size = shape[dim];
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let outer_size: usize = shape[..dim].iter().product::<usize>();
    let inner_size: usize = shape[dim + 1..].iter().product::<usize>();
    let out_size = outer_size * inner_size;
    let data: &[E] = tensor.storage();
    let start_offset = tensor.layout().start_offset();

    let find = |flat_idx: usize| -> (E, isize) {
        let outer = flat_idx / inner_size;
        let inner = flat_idx % inner_size;
        let base = start_offset + outer * dim_size * inner_size + inner;
        let mut best = to_f32(data[base]);
        let mut best_idx: isize = 0;
        for d in 1..dim_size {
            let val = to_f32(data[base + d * inner_size]);
            if is_better(val, best) {
                best = val;
                best_idx = d as isize;
            }
        }
        (from_f32(best), best_idx)
    };

    #[cfg(feature = "rayon")]
    let (values, indices): (Vec<E>, Vec<isize>) =
        if out_size * dim_size >= EXTREMUM_PARALLEL_THRESHOLD {
            (0..out_size).into_par_iter().map(&find).unzip()
        } else {
            (0..out_size).map(find).unzip()
        };

    #[cfg(not(feature = "rayon"))]
    let (values, indices): (Vec<E>, Vec<isize>) = (0..out_size).map(find).unzip();

    let val_tensor = FlexTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::from(out_shape.clone())),
        E::dtype(),
    );
    let idx_tensor = FlexTensor::new(
        Bytes::from_elems(indices),
        Layout::contiguous(Shape::from(out_shape)),
        INDEX_DTYPE,
    );
    (val_tensor, idx_tensor)
}

// ============================================================================
// Scalar division helpers
// ============================================================================

fn scalar_div<E: Element + bytemuck::Pod + core::ops::Div<Output = E> + Copy>(
    mut tensor: FlexTensor,
    divisor: E,
) -> FlexTensor {
    let data: &mut [E] = tensor.storage_mut();
    for x in data.iter_mut() {
        *x = *x / divisor;
    }
    tensor
}

// ============================================================================
// Tests
// ============================================================================

// Tests kept here exercise flex-specific behavior: dtype storage selection
// for every numeric width (i8/i16/i32/i64/u8/u16/u32/u64, bf16/f16/f32/f64)
// and edge cases of the dtype-specific reduction kernels (zero-sized
// non-reduced dims across contiguous/half/widening paths, dim sizes that
// exceed the element's max, f16 mean overflow fusion, zero-size panics on
// max_dim/min_dim). Plain 1d/2d smokes, NaN propagation, negative-stride
// (flipped/transposed/narrowed) variants, 4D middle-dim reductions, and
// permuted-argmax regressions have been migrated to burn-backend-tests so
// they run against every backend. When adding new tests, keep them here
// only if they probe flex dtype storage or panic on flex internals;
// otherwise add them to crates/burn-backend-tests/tests/tensor/{float,int}/ops/.
#[cfg(test)]
mod tests {
    use alloc::vec;
    use burn_backend::TensorData;
    use burn_backend::ops::{FloatTensorOps, IntTensorOps};
    use burn_std::{bf16, f16};

    use crate::{Flex, FlexTensor};

    #[test]
    fn test_mean_f16_overflow_intermediate_sum() {
        // Scalar `mean()` for f16 must fuse sum+divide on the f32 accumulator.
        // Sum of 0..1024 is 523776, well above f16::MAX (65504), so a naive
        // sum-then-divide that materialises the intermediate in f16 would clip
        // to inf. The final mean (511.5) fits f16 comfortably.
        let data: Vec<f16> = (0..1024).map(|i| f16::from_f32(i as f32)).collect();
        let tensor = FlexTensor::from_data(TensorData::new(data, [1024]));

        let result = Flex::float_mean(tensor);
        let result_data = result.into_data();
        let values: &[f16] = bytemuck::cast_slice(&result_data.bytes);

        assert_eq!(values.len(), 1);
        let mean = values[0].to_f32();
        assert!(mean.is_finite(), "mean overflowed to {mean}");
        assert!((mean - 511.5).abs() < 0.5, "expected ~511.5, got {mean}");
    }

    #[test]
    fn test_mean_dim_f16_zero_outer_dim() {
        // Regression test for mean_dim_half / sum_dim_contiguous_f32 clamping
        // `outer_size.max(1)` in the last-dim branch: shape [0, 4] reducing
        // dim=1 has outer_size=0, and the old clamp produced rows=1 which
        // ran sum_rows_f32 past the end of an empty buffer. Result should be
        // an empty tensor with shape [0, 1].
        let data: Vec<f16> = Vec::new();
        let tensor = FlexTensor::from_data(TensorData::new(data, [0, 4]));
        let result = Flex::float_mean_dim(tensor, 1);

        assert_eq!(result.layout().shape().to_vec(), vec![0, 1]);
        let result_data = result.into_data();
        let values: &[f16] = bytemuck::cast_slice(&result_data.bytes);
        assert!(values.is_empty());
    }

    #[test]
    fn test_sum_dim_f32_zero_outer_dim() {
        // Regression: the f32 last-dim SIMD path (`reduce_last_dim_f32`) used
        // `rows = outer_size.max(1)` which would index past the end of an
        // empty data buffer for shape [0, K]. Guarded by the `out_size == 0`
        // early return in `reduce_dim_f32`.
        let data: Vec<f32> = Vec::new();
        let tensor = FlexTensor::from_data(TensorData::new(data, [0, 4]));
        let result = Flex::float_sum_dim(tensor, 1);

        assert_eq!(result.layout().shape().to_vec(), vec![0, 1]);
        assert!(result.into_data().bytes.is_empty());
    }

    #[test]
    fn test_sum_dim_f32_zero_inner_dim() {
        // Mirror of the outer-zero case: shape [3, 0] reducing dim=0 has
        // inner_size=0.
        let data: Vec<f32> = Vec::new();
        let tensor = FlexTensor::from_data(TensorData::new(data, [3, 0]));
        let result = Flex::float_sum_dim(tensor, 0);

        assert_eq!(result.layout().shape().to_vec(), vec![1, 0]);
        assert!(result.into_data().bytes.is_empty());
    }

    #[test]
    fn test_sum_dim_f64_zero_outer_dim() {
        // Covers `reduce_dim_impl` (generic contiguous/non-contiguous path)
        // for the zero-sized non-reduced dim case.
        let data: Vec<f64> = Vec::new();
        let tensor = FlexTensor::from_data(TensorData::new(data, [0, 4]));
        let result = Flex::float_sum_dim(tensor, 1);

        assert_eq!(result.layout().shape().to_vec(), vec![0, 1]);
        assert!(result.into_data().bytes.is_empty());
    }

    #[test]
    fn test_sum_dim_i8_zero_outer_dim() {
        // Covers `reduce_dim_widening` (i8/i16/u8/u16 path that accumulates
        // in i64) for the zero-sized non-reduced dim case.
        let data: Vec<i8> = Vec::new();
        let tensor = FlexTensor::from_data(TensorData::new(data, [0, 4]));
        let result = Flex::int_sum_dim(tensor, 1);

        assert_eq!(result.layout().shape().to_vec(), vec![0, 1]);
        assert!(result.into_data().bytes.is_empty());
    }

    #[test]
    fn test_sum_dim_bf16_zero_outer_dim() {
        // Covers `reduce_dim_half` (bf16/f16 sum_dim/prod_dim path) for the
        // zero-sized non-reduced dim case.
        let data: Vec<bf16> = Vec::new();
        let tensor = FlexTensor::from_data(TensorData::new(data, [0, 4]));
        let result = Flex::float_sum_dim(tensor, 1);

        assert_eq!(result.layout().shape().to_vec(), vec![0, 1]);
        assert!(result.into_data().bytes.is_empty());
    }

    #[test]
    fn test_mean_dim_i8_large_dimension() {
        // dim_size=200 exceeds i8::MAX (127). Before the fix, 200 as i8 = -56,
        // causing wrong results (or 256 as i8 = 0 causing div-by-zero).
        let mut data: Vec<i8> = vec![0i8; 200];
        data[0] = 100;
        let tensor = FlexTensor::from_data(TensorData::new(data, [1, 200]));
        let result = Flex::int_mean_dim(tensor, 1);

        let result_data = result.into_data();
        let values: Vec<i8> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        // integer division: 100 / 200 = 0
        assert_eq!(values, vec![0]);
    }

    #[test]
    fn test_mean_dim_i16_large_dimension() {
        // dim_size=40000 exceeds i16::MAX (32767).
        let mut data: Vec<i16> = vec![0i16; 40000];
        data[0] = 32000;
        let tensor = FlexTensor::from_data(TensorData::new(data, [1, 40000]));
        let result = Flex::int_mean_dim(tensor, 1);

        let result_data = result.into_data();
        let values: Vec<i16> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![0]);
    }

    #[test]
    fn test_sum_i32() {
        let data: Vec<i32> = vec![1, 2, 3, 4, 5];
        let tensor = FlexTensor::from_data(TensorData::new(data, [5]));
        let result = Flex::int_sum(tensor);

        assert_eq!(result.layout().shape().to_vec(), vec![1]);
        let result_data = result.into_data();
        let values: Vec<i32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![15]);
    }

    #[test]
    fn test_sum_dim_i32() {
        let data: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3]));
        let result = Flex::int_sum_dim(tensor, 1);

        assert_eq!(result.layout().shape().to_vec(), vec![2, 1]);
        let result_data = result.into_data();
        let values: Vec<i32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![6, 15]);
    }

    #[test]
    fn test_argmax_i32() {
        let data: Vec<i32> = vec![1, 5, 3, 2, 4];
        let tensor = FlexTensor::from_data(TensorData::new(data, [5]));
        let result = Flex::int_argmax(tensor, 0);

        assert_eq!(result.layout().shape().to_vec(), vec![1]);
        let result_data = result.into_data();
        #[cfg(target_pointer_width = "64")]
        let values: Vec<i64> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        #[cfg(target_pointer_width = "32")]
        let values: Vec<i64> = bytemuck::cast_slice::<u8, i32>(&result_data.bytes)
            .iter()
            .map(|&v| v as i64)
            .collect();
        assert_eq!(values, vec![1]);
    }

    #[test]
    #[should_panic(expected = "dimension 0 has size 0")]
    fn test_max_dim_zero_size_panics() {
        let tensor = FlexTensor::from_data(TensorData::new(Vec::<f32>::new(), [0, 3]));
        Flex::float_max_dim(tensor, 0);
    }

    #[test]
    #[should_panic(expected = "dimension 1 has size 0")]
    fn test_min_dim_zero_size_panics() {
        let tensor = FlexTensor::from_data(TensorData::new(Vec::<f32>::new(), [3, 0]));
        Flex::float_min_dim(tensor, 1);
    }

    // === Unsigned integer dtype tests ===

    #[test]
    fn test_sum_u32() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![10u32, 20, 30], [3]));
        let result = Flex::int_sum(tensor);
        let data: Vec<u32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![60]);
    }

    #[test]
    fn test_sum_u64() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![100u64, 200, 300], [3]));
        let result = Flex::int_sum(tensor);
        let data: Vec<u64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![600]);
    }

    #[test]
    fn test_sum_dim_u8() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![1u8, 2, 3, 4], [2, 2]));
        let result = Flex::int_sum_dim(tensor, 1);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3, 7]);
    }

    #[test]
    fn test_prod_u16() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![2u16, 3, 5], [3]));
        let result = Flex::int_prod(tensor);
        let data: Vec<u16> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![30]);
    }

    #[test]
    fn test_max_u32() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![5u32, 100, 42], [3]));
        let result = Flex::int_max(tensor);
        let data: Vec<u32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![100]);
    }

    #[test]
    fn test_min_u8() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![5u8, 1, 42], [3]));
        let result = Flex::int_min(tensor);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1]);
    }

    #[test]
    fn test_max_dim_u64() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![10u64, 20, 30, 5], [2, 2]));
        let result = Flex::int_max_dim(tensor, 1);
        let data: Vec<u64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![20, 30]);
    }

    #[test]
    fn test_min_dim_u16() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![10u16, 2, 30, 5], [2, 2]));
        let result = Flex::int_min_dim(tensor, 1);
        let data: Vec<u16> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![2, 5]);
    }

    #[test]
    fn test_mean_dim_u8() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![10u8, 20, 30, 40], [2, 2]));
        let result = Flex::int_mean_dim(tensor, 1);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![15, 35]);
    }

    #[test]
    fn test_max_dim_with_indices_u32() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![5u32, 10, 3, 8], [2, 2]));
        let (values, indices) = Flex::int_max_dim_with_indices(tensor, 1);
        let vals: Vec<u32> = values.into_data().to_vec().unwrap();
        let idxs: Vec<isize> = bytemuck::cast_slice(&indices.into_data().bytes).to_vec();
        assert_eq!(vals, vec![10, 8]);
        assert_eq!(idxs, vec![1, 1]);
    }
}
