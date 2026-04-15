//! Slice operations for FlexTensor.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, Slice, bf16, f16};

use crate::{FlexTensor, Layout};

/// Slice a tensor according to the given slice parameters.
///
/// For positive steps, this is zero-copy (metadata only).
/// For negative steps, data is copied to handle the reversal.
pub fn slice(tensor: FlexTensor, slices: &[Slice]) -> FlexTensor {
    let (new_layout, needs_copy) = tensor.layout().slice(slices);

    if !needs_copy {
        // Zero-copy: share data with new layout
        FlexTensor::from_arc(tensor.data_arc(), new_layout, tensor.dtype())
    } else {
        // Needs copy due to negative steps
        slice_with_copy(&tensor, slices)
    }
}

/// Slice with data copy (handles negative steps).
fn slice_with_copy(tensor: &FlexTensor, slices: &[Slice]) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => slice_copy_impl::<f32>(tensor, slices),
        DType::F64 => slice_copy_impl::<f64>(tensor, slices),
        DType::F16 => slice_copy_impl::<f16>(tensor, slices),
        DType::BF16 => slice_copy_impl::<bf16>(tensor, slices),
        DType::I32 => slice_copy_impl::<i32>(tensor, slices),
        DType::I64 => slice_copy_impl::<i64>(tensor, slices),
        DType::I16 => slice_copy_impl::<i16>(tensor, slices),
        DType::I8 => slice_copy_impl::<i8>(tensor, slices),
        DType::U32 => slice_copy_impl::<u32>(tensor, slices),
        DType::U64 => slice_copy_impl::<u64>(tensor, slices),
        DType::U16 => slice_copy_impl::<u16>(tensor, slices),
        DType::U8 => slice_copy_impl::<u8>(tensor, slices),
        DType::Bool(_) => slice_copy_impl::<u8>(tensor, slices),
        _ => panic!("slice: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Generic slice implementation with copy.
fn slice_copy_impl<E: Element + bytemuck::Pod + Default>(
    tensor: &FlexTensor,
    slices: &[Slice],
) -> FlexTensor {
    let src = tensor.storage::<E>();
    let src_layout = tensor.layout();
    let ndims = src_layout.num_dims();

    // Calculate output shape and collect normalized slice info
    let mut out_shape = Vec::with_capacity(ndims);
    let mut slice_info: Vec<(usize, usize, isize)> = Vec::with_capacity(ndims); // (start, len, step)

    for dim in 0..ndims {
        let dim_size = src_layout.shape()[dim] as isize;

        let slice = if dim < slices.len() {
            &slices[dim]
        } else {
            // Default: full range
            &Slice::new(0, None, 1)
        };

        let (start, len, step) = compute_slice_info(slice, dim_size);
        out_shape.push(len);
        slice_info.push((start, len, step));
    }

    let out_layout = Layout::contiguous(Shape::from(out_shape.clone()));
    let num_elements = out_layout.num_elements();

    if num_elements == 0 {
        let bytes = Bytes::from_elems::<E>(Vec::new());
        return FlexTensor::new(bytes, out_layout, tensor.dtype());
    }

    // Allocate output
    let mut out_data: Vec<E> = Vec::with_capacity(num_elements);

    // Use recursive iteration for arbitrary dimensions
    let mut indices = vec![0usize; ndims];
    copy_slice_recursive(src, src_layout, &slice_info, &mut out_data, &mut indices, 0);

    let bytes = Bytes::from_elems(out_data);
    FlexTensor::new(bytes, out_layout, tensor.dtype())
}

/// Recursively copy sliced elements.
fn copy_slice_recursive<E: Copy>(
    src: &[E],
    src_layout: &Layout,
    slice_info: &[(usize, usize, isize)],
    out: &mut Vec<E>,
    indices: &mut [usize],
    dim: usize,
) {
    let ndims = src_layout.num_dims();

    if dim == ndims {
        // Base case: copy single element
        let src_idx = compute_src_index(src_layout, slice_info, indices);
        out.push(src[src_idx]);
        return;
    }

    let (_, len, _) = slice_info[dim];

    for i in 0..len {
        indices[dim] = i;
        copy_slice_recursive(src, src_layout, slice_info, out, indices, dim + 1);
    }
}

/// Compute source index from output indices and slice info.
fn compute_src_index(
    layout: &Layout,
    slice_info: &[(usize, usize, isize)],
    out_indices: &[usize],
) -> usize {
    let mut idx = layout.start_offset() as isize;
    for (dim, &out_i) in out_indices.iter().enumerate() {
        let (start, _, step) = slice_info[dim];
        let src_i = if step > 0 {
            start + out_i * step as usize
        } else {
            // Negative step: start from high index, go down
            let result = start as isize - (out_i as isize) * (-step);
            debug_assert!(result >= 0, "slice: negative source index at dim {dim}");
            result as usize
        };
        idx += src_i as isize * layout.strides()[dim];
    }
    debug_assert!(idx >= 0, "slice: negative final index");
    idx as usize
}

/// Normalize a potentially negative index to a positive one.
fn normalize_index(idx: isize, dim_size: isize) -> usize {
    if idx < 0 {
        (dim_size + idx).max(0) as usize
    } else {
        idx as usize
    }
}

/// Assign values to a slice of a tensor.
pub fn slice_assign(tensor: FlexTensor, slices: &[Slice], value: FlexTensor) -> FlexTensor {
    match tensor.dtype() {
        DType::F32 => slice_assign_impl::<f32>(tensor, slices, value),
        DType::F64 => slice_assign_impl::<f64>(tensor, slices, value),
        DType::F16 => slice_assign_impl::<f16>(tensor, slices, value),
        DType::BF16 => slice_assign_impl::<bf16>(tensor, slices, value),
        DType::I32 => slice_assign_impl::<i32>(tensor, slices, value),
        DType::I64 => slice_assign_impl::<i64>(tensor, slices, value),
        DType::I16 => slice_assign_impl::<i16>(tensor, slices, value),
        DType::I8 => slice_assign_impl::<i8>(tensor, slices, value),
        DType::U32 => slice_assign_impl::<u32>(tensor, slices, value),
        DType::U64 => slice_assign_impl::<u64>(tensor, slices, value),
        DType::U16 => slice_assign_impl::<u16>(tensor, slices, value),
        DType::U8 => slice_assign_impl::<u8>(tensor, slices, value),
        DType::Bool(_) => slice_assign_impl::<u8>(tensor, slices, value),
        _ => panic!("slice_assign: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Generic slice assign implementation.
fn slice_assign_impl<E: Element + bytemuck::Pod + Clone>(
    tensor: FlexTensor,
    slices: &[Slice],
    value: FlexTensor,
) -> FlexTensor {
    // Broadcast-scalar fast path: if `value` is a fully-broadcast
    // scalar (all strides zero), read the scalar once instead of
    // materializing the expansion via `to_contiguous`. The
    // `num_elements > 0` gate also guards the storage read against
    // zero-sized sources where `iter().all(...)` would be vacuously
    // true.
    if value.layout().num_elements() > 0 && value.layout().strides().iter().all(|&s| s == 0) {
        let scalar = value.storage::<E>()[value.layout().start_offset()];
        return slice_write_impl::<E>(tensor, slices, WriteSource::Scalar(scalar));
    }

    let value = value.to_contiguous();
    let val_src: &[E] = value.storage::<E>();
    slice_write_impl::<E>(tensor, slices, WriteSource::Slice(val_src))
}

/// Source to splat into a sliced region of a destination tensor. The
/// two variants drive the same dispatch tree; `Scalar` hits the
/// broadcast-scalar fast path (no value buffer), `Slice` hits the
/// memcpy-style assign path (advances through `val_src`).
#[derive(Copy, Clone)]
enum WriteSource<'a, E: Copy> {
    Scalar(E),
    Slice(&'a [E]),
}

impl<'a, E: Copy> WriteSource<'a, E> {
    /// Write a contiguous span of `dst` starting at `dst_offset` for
    /// `len` elements. For `Slice`, `src_offset` is the current
    /// position in the value buffer.
    #[inline]
    fn write_span(self, dst: &mut [E], dst_offset: usize, len: usize, src_offset: usize) {
        match self {
            WriteSource::Scalar(s) => dst[dst_offset..dst_offset + len].fill(s),
            WriteSource::Slice(src) => dst[dst_offset..dst_offset + len]
                .copy_from_slice(&src[src_offset..src_offset + len]),
        }
    }

    /// Write a single element. `src_idx` is only read in the `Slice`
    /// variant.
    #[inline]
    fn write_element(self, dst: &mut [E], dst_idx: usize, src_idx: usize) {
        match self {
            WriteSource::Scalar(s) => dst[dst_idx] = s,
            WriteSource::Slice(src) => dst[dst_idx] = src[src_idx],
        }
    }
}

/// Unified slice writer used by both `slice_assign_impl` and the
/// scalar-broadcast fast path. Walks the destination's sliced region
/// (1D / 2D inner-contig / ND inner-contig / strided fallback) and
/// pulls values from the given [`WriteSource`].
fn slice_write_impl<E: Element + bytemuck::Pod>(
    tensor: FlexTensor,
    slices: &[Slice],
    source: WriteSource<'_, E>,
) -> FlexTensor {
    let mut tensor = tensor.to_contiguous();
    let dst_layout = tensor.layout().clone();
    let ndims = dst_layout.num_dims();

    let slice_info: Vec<(usize, usize, isize)> = (0..ndims)
        .map(|dim| {
            let dim_size = dst_layout.shape()[dim] as isize;
            let slice = if dim < slices.len() {
                &slices[dim]
            } else {
                &Slice::new(0, None, 1)
            };
            compute_slice_info(slice, dim_size)
        })
        .collect();

    let dst = tensor.storage_mut::<E>();

    let inner_contiguous = slice_info
        .last()
        .map(|(_, _, step)| *step == 1)
        .unwrap_or(false);

    if ndims == 0 {
        // Rank 0: single scalar destination. Only reachable from the
        // scalar fast path; `slice_assign` on a rank-0 tensor with a
        // rank-0 source also ends up here.
        if !dst.is_empty() {
            source.write_element(dst, 0, 0);
        }
    } else if ndims == 1 {
        let (start, len, step) = slice_info[0];
        if step == 1 {
            source.write_span(dst, start, len, 0);
        } else {
            for i in 0..len {
                let dst_i = if step > 0 {
                    start + i * step as usize
                } else {
                    (start as isize - (i as isize) * (-step)) as usize
                };
                source.write_element(dst, dst_i, i);
            }
        }
    } else if ndims == 2 && inner_contiguous {
        let (row_start, row_len, row_step) = slice_info[0];
        let (col_start, col_len, _) = slice_info[1];
        let dst_cols = dst_layout.shape()[1];

        let mut val_offset = 0;
        for r in 0..row_len {
            let row_idx = if row_step > 0 {
                row_start + r * row_step as usize
            } else {
                (row_start as isize - (r as isize) * (-row_step)) as usize
            };
            let dst_row_start = row_idx * dst_cols + col_start;
            source.write_span(dst, dst_row_start, col_len, val_offset);
            val_offset += col_len;
        }
    } else if inner_contiguous {
        let inner_len = slice_info[ndims - 1].1;
        let outer_dims = ndims - 1;
        let dst_strides = dst_layout.strides();

        let outer_count: usize = slice_info.iter().take(outer_dims).map(|i| i.1).product();

        let mut outer_indices = vec![0usize; outer_dims];
        let mut val_offset = 0;

        for _ in 0..outer_count {
            let mut dst_offset = dst_layout.start_offset() as isize;
            for (dim, &idx) in outer_indices.iter().enumerate() {
                let (start, _, step) = slice_info[dim];
                let src_i = if step > 0 {
                    start + idx * step as usize
                } else {
                    (start as isize - (idx as isize) * (-step)) as usize
                };
                dst_offset += src_i as isize * dst_strides[dim];
            }
            dst_offset += slice_info[ndims - 1].0 as isize * dst_strides[ndims - 1];
            let dst_offset = dst_offset as usize;

            source.write_span(dst, dst_offset, inner_len, val_offset);
            val_offset += inner_len;

            // Odometer increment over outer dims.
            for dim in (0..outer_dims).rev() {
                outer_indices[dim] += 1;
                if outer_indices[dim] < slice_info[dim].1 {
                    break;
                }
                outer_indices[dim] = 0;
            }
        }
    } else {
        let total_elements: usize = slice_info.iter().map(|(_, len, _)| len).product();
        let dst_strides = dst_layout.strides();
        let mut indices = vec![0usize; ndims];

        for i in 0..total_elements {
            let mut dst_offset = dst_layout.start_offset() as isize;
            for (dim, &idx) in indices.iter().enumerate() {
                let (start, _, step) = slice_info[dim];
                let src_i = if step > 0 {
                    start + idx * step as usize
                } else {
                    (start as isize - (idx as isize) * (-step)) as usize
                };
                dst_offset += src_i as isize * dst_strides[dim];
            }

            source.write_element(dst, dst_offset as usize, i);

            for dim in (0..ndims).rev() {
                indices[dim] += 1;
                if indices[dim] < slice_info[dim].1 {
                    break;
                }
                indices[dim] = 0;
            }
        }
    }

    tensor
}

/// Compute slice info (start, len, step) for a dimension.
/// For negative step: start is the LAST index in the range (end-1), iterating down.
fn compute_slice_info(slice: &Slice, dim_size: isize) -> (usize, usize, isize) {
    let step = slice.step;
    let abs_step = step.unsigned_abs();

    // Normalize start and end to [0, dim_size]
    let range_start = normalize_index(slice.start, dim_size);
    let range_end = match slice.end {
        Some(e) => normalize_index(e, dim_size).min(dim_size as usize),
        None => dim_size as usize,
    };

    let len = if range_end > range_start {
        (range_end - range_start).div_ceil(abs_step)
    } else {
        0
    };

    if step > 0 {
        // Forward: start at low index, go up
        (range_start, len, step)
    } else {
        // Reverse: start at end-1 (highest index in range), go down
        // For s![2..8;-2]: start from index 7, go to 5, then 3
        let reverse_start = if range_end > range_start {
            range_end - 1
        } else {
            range_start
        };
        (reverse_start, len, step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_slice_basic() {
        // Create a 2x3 tensor: [[0, 1, 2], [3, 4, 5]]
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3]));

        // Slice [0:1, 1:3] -> [[1, 2]]
        let slices = vec![Slice::new(0, Some(1), 1), Slice::new(1, Some(3), 1)];
        let result = slice(tensor, &slices);

        assert_eq!(result.layout().shape().to_vec(), vec![1, 2]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_slice_with_step() {
        // Create a 1D tensor: [0, 1, 2, 3, 4, 5]
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, [6]));

        // Slice [0:6:2] -> [0, 2, 4]
        let slices = vec![Slice::new(0, Some(6), 2)];
        let result = slice(tensor, &slices);

        assert_eq!(result.layout().shape().to_vec(), vec![3]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_slice_negative_index() {
        // Create a 1D tensor: [0, 1, 2, 3, 4]
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, [5]));

        // Slice [-3:] -> [2, 3, 4]
        let slices = vec![Slice::new(-3, None, 1)];
        let result = slice(tensor, &slices);

        assert_eq!(result.layout().shape().to_vec(), vec![3]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_slice_negative_step() {
        // Create a 1D tensor: [0, 1, 2, 3, 4]
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, [5]));

        // Slice [0..;-1] -> [4, 3, 2, 1, 0] (reverse full range)
        // In Burn's semantics: range selects elements, step determines order
        let slices = vec![Slice::new(0, None, -1)];
        let result = slice(tensor, &slices);

        assert_eq!(result.layout().shape().to_vec(), vec![5]);
        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![4.0, 3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_slice_assign_1d() {
        // Create a 1D tensor: [0, 1, 2, 3, 4]
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, [5]));

        // Assign [10, 11, 12] to positions [1:4]
        let value_data: Vec<f32> = vec![10.0, 11.0, 12.0];
        let value = FlexTensor::from_data(TensorData::new(value_data, [3]));
        let slices = vec![Slice::new(1, Some(4), 1)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![0.0, 10.0, 11.0, 12.0, 4.0]);
    }

    #[test]
    fn test_slice_assign_2d() {
        // Create a 3x3 tensor
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, [3, 3]));

        // Assign [[10, 11], [12, 13]] to [1:3, 1:3]
        let value_data: Vec<f32> = vec![10.0, 11.0, 12.0, 13.0];
        let value = FlexTensor::from_data(TensorData::new(value_data, [2, 2]));
        let slices = vec![Slice::new(1, Some(3), 1), Slice::new(1, Some(3), 1)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(
            values,
            vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 6.0, 12.0, 13.0,]
        );
    }

    #[test]
    fn test_slice_assign_2d_full_row() {
        // Create a 3x4 tensor
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensor = FlexTensor::from_data(TensorData::new(data, [3, 4]));

        // Assign [100, 101, 102, 103] to row 1
        let value_data: Vec<f32> = vec![100.0, 101.0, 102.0, 103.0];
        let value = FlexTensor::from_data(TensorData::new(value_data, [1, 4]));
        let slices = vec![Slice::new(1, Some(2), 1), Slice::new(0, None, 1)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(
            values,
            vec![
                0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0, 8.0, 9.0, 10.0, 11.0,
            ]
        );
    }

    // Broadcast-scalar fast path tests: mimic what
    // `Tensor::slice_fill` produces (a 1-element source expanded to
    // the slice shape with all strides zero).

    fn broadcast_scalar_f32(value: f32, target_shape: &[usize]) -> FlexTensor {
        let scalar_tensor = FlexTensor::from_data(TensorData::new(vec![value], [1]));
        crate::ops::expand::expand(scalar_tensor, Shape::from(target_shape.to_vec()))
    }

    #[test]
    fn test_slice_assign_broadcast_scalar_1d_contiguous() {
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let tensor = FlexTensor::from_data(TensorData::new(data, [5]));
        let value = broadcast_scalar_f32(7.0, &[3]);
        let slices = vec![Slice::new(1, Some(4), 1)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![0.0, 7.0, 7.0, 7.0, 4.0]);
    }

    #[test]
    fn test_slice_assign_broadcast_scalar_2d_inner_contiguous() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tensor = FlexTensor::from_data(TensorData::new(data, [4, 4]));
        let value = broadcast_scalar_f32(-1.0, &[2, 2]);
        let slices = vec![Slice::new(1, Some(3), 1), Slice::new(1, Some(3), 1)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(
            values,
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, -1.0, -1.0, 7.0, 8.0, -1.0, -1.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ]
        );
    }

    #[test]
    fn test_slice_assign_broadcast_scalar_3d_inner_contiguous() {
        // 3D case that matched the user-reported regression shape.
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3, 4]));
        let value = broadcast_scalar_f32(9.0, &[1, 2, 2]);
        let slices = vec![
            Slice::new(0, Some(1), 1),
            Slice::new(0, Some(2), 1),
            Slice::new(1, Some(3), 1),
        ];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        // Region [0..1, 0..2, 1..3] fills positions (0, 0, 1), (0, 0, 2),
        // (0, 1, 1), (0, 1, 2) with 9.0. Linear indices: 1, 2, 5, 6.
        let mut expected: Vec<f32> = (0..24).map(|i| i as f32).collect();
        for &i in &[1usize, 2, 5, 6] {
            expected[i] = 9.0;
        }
        assert_eq!(values, expected);
    }

    #[test]
    fn test_slice_assign_broadcast_scalar_strided_fallback() {
        // Stepped slice: hits the strided fallback (not inner-contiguous).
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let tensor = FlexTensor::from_data(TensorData::new(data, [10]));
        // Source needs to match slice_info length: s![0..10;2] selects
        // 5 positions (0, 2, 4, 6, 8), so the expand target is [5].
        let value = broadcast_scalar_f32(0.0, &[5]);
        let slices = vec![Slice::new(0, Some(10), 2)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(
            values,
            vec![0.0, 1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0]
        );
    }

    /// Broadcast-scalar fast path on a non-f32 dtype.
    #[test]
    fn test_slice_assign_broadcast_scalar_i64() {
        fn broadcast_scalar_i64(value: i64, target_shape: &[usize]) -> FlexTensor {
            let scalar_tensor = FlexTensor::from_data(TensorData::new(vec![value], [1]));
            crate::ops::expand::expand(scalar_tensor, Shape::from(target_shape.to_vec()))
        }

        let data: Vec<i64> = (0..12).collect();
        let tensor = FlexTensor::from_data(TensorData::new(data, [3, 4]));
        let value = broadcast_scalar_i64(-7, &[2, 2]);
        let slices = vec![Slice::new(0, Some(2), 1), Slice::new(1, Some(3), 1)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<i64> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        // Positions (0, 1), (0, 2), (1, 1), (1, 2): linear indices 1,
        // 2, 5, 6 get replaced by -7.
        assert_eq!(values, vec![0, -7, -7, 3, 4, -7, -7, 7, 8, 9, 10, 11]);
    }

    /// ND strided fallback path: 3D slice with a stepped inner dim.
    #[test]
    fn test_slice_assign_broadcast_scalar_nd_strided_fallback() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = FlexTensor::from_data(TensorData::new(data, [2, 3, 4]));
        // Step 2 on the innermost dim means slice_info's last step != 1,
        // so inner_contiguous is false and we take the ND fallback.
        let value = broadcast_scalar_f32(9.0, &[2, 3, 2]);
        let slices = vec![
            Slice::new(0, Some(2), 1),
            Slice::new(0, Some(3), 1),
            Slice::new(0, Some(4), 2),
        ];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        // Step-2 on dim 2 picks indices 0 and 2, so within each
        // `[b, r, :]` row the 0 and 2 positions get 9.0.
        let mut expected: Vec<f32> = (0..24).map(|i| i as f32).collect();
        for b in 0..2 {
            for r in 0..3 {
                for c in [0, 2] {
                    expected[b * 12 + r * 4 + c] = 9.0;
                }
            }
        }
        assert_eq!(values, expected);
    }

    /// 2D inner-contig branch with `row_step > 1`.
    #[test]
    fn test_slice_assign_broadcast_scalar_2d_stepped_rows() {
        let data: Vec<f32> = (0..25).map(|i| i as f32).collect();
        let tensor = FlexTensor::from_data(TensorData::new(data, [5, 5]));
        // Step-2 rows pick out rows 0, 2, 4 (3 rows). Slice the inner
        // dim as a contiguous range so the 2D-inner-contig branch with
        // row_step != 1 fires.
        let value = broadcast_scalar_f32(-1.0, &[3, 3]);
        let slices = vec![Slice::new(0, Some(5), 2), Slice::new(1, Some(4), 1)];
        let result = slice_assign(tensor, &slices, value);

        let result_data = result.into_data();
        let values: Vec<f32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        let mut expected: Vec<f32> = (0..25).map(|i| i as f32).collect();
        for r in [0, 2, 4] {
            for c in 1..4 {
                expected[r * 5 + c] = -1.0;
            }
        }
        assert_eq!(values, expected);
    }
}
