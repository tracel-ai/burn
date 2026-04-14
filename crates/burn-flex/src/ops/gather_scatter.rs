//! Gather and scatter operations for indexed tensor access.

use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape};
use bytemuck::Pod;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::{FlexTensor, Layout};

/// Read indices from a tensor as `isize`, the native offset type used by the
/// gather/scatter/select kernels in this module.
///
/// This is the internal index layer for burn-flex: every indexed op
/// ([`gather`], [`scatter_add`], [`select`], [`select_add`], and the
/// [`scatter_min`]/[`scatter_max`] variants) routes its index tensor through
/// this helper before touching the element buffer. Normalising to `isize`
/// lets the kernels use a single inner-loop signature regardless of how the
/// caller's index tensor was dtyped.
///
/// # Accepted widths
///
/// Any of the integer DTypes `I8`, `I16`, `I32`, `I64`, `U8`, `U16`, `U32`,
/// `U64` is accepted. This is intentional: burn-flex's default `IntElem` is
/// I32 rather than the I64 convention used by other backends, and users can
/// also pin index tensors to any width they want via
/// `Tensor::from_data(.., (&device, DType::Ix))`. Whichever width lands here
/// is converted to `isize` on the fly.
///
/// # Zero-copy vs. owned
///
/// The return type is `Cow<'_, [isize]>` because only one width is zero-copy:
/// the one matching the host pointer width. On 64-bit targets, I64 indices
/// can be borrowed directly via `bytemuck::cast_slice` (both are 8 bytes).
/// Every other width requires an owned `Vec<isize>` with an element-wise
/// cast. U64 indices additionally go through a `try_from` to surface values
/// that would wrap when cast to `isize`.
///
/// # History
///
/// Earlier versions of `int_gather`, `int_scatter_add`, `int_select`, and
/// `int_select_add` carried a `debug_assert_eq!(indices.dtype(), DType::I64,
/// ..)` that contradicted this helper's contract. The asserts were dropped
/// in tracel-ai/burn#4776 once it was confirmed that `read_indices` had
/// always handled every supported width correctly at runtime. If you're
/// tempted to re-add a dtype check here, don't - the float siblings
/// ([`gather_f32`], [`select_f32`], ...) already share this helper without a
/// check, and asymmetry between the int and float paths was what surfaced
/// the bug.
fn read_indices(tensor: &FlexTensor) -> Cow<'_, [isize]> {
    match tensor.dtype() {
        #[cfg(target_pointer_width = "64")]
        DType::I64 => {
            const { assert!(size_of::<i64>() == size_of::<isize>()) };
            let data = tensor.storage::<i64>();
            Cow::Borrowed(bytemuck::cast_slice(data))
        }
        #[cfg(target_pointer_width = "32")]
        DType::I64 => Cow::Owned(
            tensor
                .storage::<i64>()
                .iter()
                .map(|&v| {
                    isize::try_from(v).unwrap_or_else(|_| {
                        panic!("read_indices: i64 index {v} out of isize range")
                    })
                })
                .collect(),
        ),
        #[cfg(target_pointer_width = "64")]
        DType::I32 => Cow::Owned(
            tensor
                .storage::<i32>()
                .iter()
                .map(|&v| v as isize)
                .collect(),
        ),
        #[cfg(target_pointer_width = "32")]
        DType::I32 => {
            const { assert!(size_of::<i32>() == size_of::<isize>()) };
            let data = tensor.storage::<i32>();
            Cow::Borrowed(bytemuck::cast_slice(data))
        }
        DType::I16 => Cow::Owned(
            tensor
                .storage::<i16>()
                .iter()
                .map(|&v| v as isize)
                .collect(),
        ),
        DType::I8 => Cow::Owned(tensor.storage::<i8>().iter().map(|&v| v as isize).collect()),
        DType::U64 => Cow::Owned(
            tensor
                .storage::<u64>()
                .iter()
                .map(|&v| {
                    isize::try_from(v).unwrap_or_else(|_| {
                        panic!("read_indices: u64 index {v} out of isize range")
                    })
                })
                .collect(),
        ),
        #[cfg(target_pointer_width = "64")]
        DType::U32 => Cow::Owned(
            tensor
                .storage::<u32>()
                .iter()
                .map(|&v| v as isize)
                .collect(),
        ),
        #[cfg(target_pointer_width = "32")]
        DType::U32 => Cow::Owned(
            tensor
                .storage::<u32>()
                .iter()
                .map(|&v| {
                    isize::try_from(v).unwrap_or_else(|_| {
                        panic!("read_indices: u32 index {v} out of isize range")
                    })
                })
                .collect(),
        ),
        DType::U16 => Cow::Owned(
            tensor
                .storage::<u16>()
                .iter()
                .map(|&v| v as isize)
                .collect(),
        ),
        DType::U8 => Cow::Owned(tensor.storage::<u8>().iter().map(|&v| v as isize).collect()),
        other => panic!("read_indices: unsupported index dtype {:?}", other),
    }
}

#[cold]
#[inline(never)]
fn index_oob(raw: isize, dim_size: usize) -> ! {
    panic!("index {raw} out of bounds for dimension of size {dim_size}");
}

/// Validate an index is non-negative and within bounds, panicking with a clear message otherwise.
#[inline(always)]
fn checked_index(raw: isize, dim_size: usize) -> usize {
    if raw < 0 || raw as usize >= dim_size {
        index_oob(raw, dim_size);
    }
    raw as usize
}

/// Gather values from tensor along a dimension using indices.
///
/// For a 2D tensor with dim=1:
/// output[i, j] = tensor[i, indices[i, j]]
///
/// The output has the same shape as indices.
pub fn gather<E: Element + Pod + Default + Copy + Send + Sync>(
    tensor: FlexTensor,
    dim: usize,
    indices: FlexTensor,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let indices = indices.to_contiguous();

    let tensor_shape = tensor.layout().shape();
    let indices_shape = indices.layout().shape();
    let ndims = tensor_shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    // Validate shapes: all dims except `dim` must match between tensor and indices
    for i in 0..ndims {
        if i != dim {
            assert_eq!(
                tensor_shape[i], indices_shape[i],
                "gather: shape mismatch at dim {}: tensor {} vs indices {}",
                i, tensor_shape[i], indices_shape[i]
            );
        }
    }

    let tensor_data: &[E] = tensor.storage();
    let indices_data = read_indices(&indices);

    // Calculate strides for tensor (row-major)
    let tensor_strides: Vec<usize> = compute_strides(tensor_shape);
    let indices_strides: Vec<usize> = compute_strides(indices_shape);

    let output_size = indices_shape.num_elements();

    // Use specialized 2D implementation for common case
    if ndims == 2 {
        let result = gather_2d::<E>(
            tensor_data,
            &indices_data,
            tensor_shape[0],
            tensor_shape[1],
            indices_shape[0],
            indices_shape[1],
            dim,
        );
        let bytes = Bytes::from_elems(result);
        return FlexTensor::new(bytes, Layout::contiguous(indices_shape.clone()), E::dtype());
    }

    // General N-D case with pre-allocated coordinates
    let dim_stride = tensor_strides[dim];

    let gather_dim_size = tensor_shape[dim];

    #[cfg(feature = "rayon")]
    let result: Vec<E> = (0..output_size)
        .into_par_iter()
        .map(|out_idx| {
            let index_val = checked_index(indices_data[out_idx], gather_dim_size);
            let src_idx = compute_gather_index(
                out_idx,
                index_val,
                dim,
                dim_stride,
                &indices_strides,
                &tensor_strides,
                ndims,
            );
            tensor_data[src_idx]
        })
        .collect();

    #[cfg(not(feature = "rayon"))]
    let result: Vec<E> = (0..output_size)
        .map(|out_idx| {
            let index_val = checked_index(indices_data[out_idx], gather_dim_size);
            let src_idx = compute_gather_index(
                out_idx,
                index_val,
                dim,
                dim_stride,
                &indices_strides,
                &tensor_strides,
                ndims,
            );
            tensor_data[src_idx]
        })
        .collect();

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(bytes, Layout::contiguous(indices_shape.clone()), E::dtype())
}

/// Optimized 2D gather implementation.
#[inline]
fn gather_2d<E: Element + Pod + Default + Copy + Send + Sync>(
    tensor_data: &[E],
    indices_data: &[isize],
    tensor_rows: usize,
    tensor_cols: usize,
    indices_rows: usize,
    indices_cols: usize,
    dim: usize,
) -> Vec<E> {
    let output_size = indices_rows * indices_cols;
    let dim_size = if dim == 0 { tensor_rows } else { tensor_cols };

    let mut result = vec![E::default(); output_size];

    #[cfg(feature = "rayon")]
    const PARALLEL_THRESHOLD: usize = 256 * 1024;

    #[cfg(feature = "rayon")]
    if output_size >= PARALLEL_THRESHOLD {
        if dim == 0 {
            result
                .par_chunks_mut(indices_cols)
                .enumerate()
                .for_each(|(i, row)| {
                    for j in 0..indices_cols {
                        let src_row = checked_index(indices_data[i * indices_cols + j], dim_size);
                        row[j] = tensor_data[src_row * tensor_cols + j];
                    }
                });
        } else {
            result
                .par_chunks_mut(indices_cols)
                .enumerate()
                .for_each(|(i, row)| {
                    for j in 0..indices_cols {
                        let src_col = checked_index(indices_data[i * indices_cols + j], dim_size);
                        row[j] = tensor_data[i * tensor_cols + src_col];
                    }
                });
        }
    } else if dim == 0 {
        for i in 0..indices_rows {
            for j in 0..indices_cols {
                let src_row = checked_index(indices_data[i * indices_cols + j], dim_size);
                result[i * indices_cols + j] = tensor_data[src_row * tensor_cols + j];
            }
        }
    } else {
        for i in 0..indices_rows {
            for j in 0..indices_cols {
                let src_col = checked_index(indices_data[i * indices_cols + j], dim_size);
                result[i * indices_cols + j] = tensor_data[i * tensor_cols + src_col];
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        if dim == 0 {
            for i in 0..indices_rows {
                for j in 0..indices_cols {
                    let src_row = checked_index(indices_data[i * indices_cols + j], dim_size);
                    result[i * indices_cols + j] = tensor_data[src_row * tensor_cols + j];
                }
            }
        } else {
            for i in 0..indices_rows {
                for j in 0..indices_cols {
                    let src_col = checked_index(indices_data[i * indices_cols + j], dim_size);
                    result[i * indices_cols + j] = tensor_data[i * tensor_cols + src_col];
                }
            }
        }
    }

    result
}

/// Compute source index for gather operation (N-D case).
#[inline]
fn compute_gather_index(
    out_idx: usize,
    index_val: usize,
    dim: usize,
    dim_stride: usize,
    indices_strides: &[usize],
    tensor_strides: &[usize],
    ndims: usize,
) -> usize {
    let mut src_idx = index_val * dim_stride;
    let mut remaining = out_idx;

    for d in 0..ndims {
        if d != dim {
            let coord = remaining / indices_strides[d];
            remaining %= indices_strides[d];
            src_idx += coord * tensor_strides[d];
        } else {
            remaining %= indices_strides[d];
        }
    }
    src_idx
}

/// Scatter add: adds values to tensor at positions specified by indices.
pub fn scatter_add<E: Element + Pod + Default + Copy + core::ops::AddAssign + Send + Sync>(
    tensor: FlexTensor,
    dim: usize,
    indices: FlexTensor,
    value: FlexTensor,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let indices = indices.to_contiguous();
    let value = value.to_contiguous();

    let tensor_shape = tensor.layout().shape().clone();
    let indices_shape = indices.layout().shape();
    let value_shape = value.layout().shape();
    let ndims = tensor_shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );
    assert_eq!(
        indices_shape,
        value_shape,
        "scatter_add: indices shape {:?} must match value shape {:?}",
        indices_shape.to_vec(),
        value_shape.to_vec()
    );

    for i in 0..ndims {
        if i != dim {
            assert_eq!(
                tensor_shape[i], indices_shape[i],
                "scatter_add: shape mismatch at dim {}: tensor {} vs indices {}",
                i, tensor_shape[i], indices_shape[i]
            );
        }
    }

    let tensor_data: &[E] = tensor.storage();
    let indices_data = read_indices(&indices);
    let value_data: &[E] = value.storage();

    let mut result: Vec<E> = tensor_data.to_vec();

    let tensor_strides: Vec<usize> = compute_strides(&tensor_shape);
    let indices_strides: Vec<usize> = compute_strides(indices_shape);

    let num_elements = indices_shape.num_elements();

    // Use specialized 2D implementation
    if ndims == 2 {
        scatter_add_2d(
            &mut result,
            &indices_data,
            value_data,
            tensor_shape[0],
            tensor_shape[1],
            indices_shape[0],
            indices_shape[1],
            dim,
        );
    } else {
        // General N-D case (sequential due to potential index conflicts)
        let dim_stride = tensor_strides[dim];
        let scatter_dim_size = tensor_shape[dim];
        for idx in 0..num_elements {
            let index_val = checked_index(indices_data[idx], scatter_dim_size);
            let dst_idx = compute_gather_index(
                idx,
                index_val,
                dim,
                dim_stride,
                &indices_strides,
                &tensor_strides,
                ndims,
            );
            result[dst_idx] += value_data[idx];
        }
    }

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(bytes, Layout::contiguous(tensor_shape), E::dtype())
}

/// Optimized 2D scatter_add implementation.
#[inline]
#[allow(clippy::too_many_arguments)]
fn scatter_add_2d<E: Copy + core::ops::AddAssign>(
    result: &mut [E],
    indices_data: &[isize],
    value_data: &[E],
    tensor_rows: usize,
    tensor_cols: usize,
    indices_rows: usize,
    indices_cols: usize,
    dim: usize,
) {
    let dim_size = if dim == 0 { tensor_rows } else { tensor_cols };
    if dim == 0 {
        for i in 0..indices_rows {
            for j in 0..indices_cols {
                let idx = i * indices_cols + j;
                let dst_row = checked_index(indices_data[idx], dim_size);
                result[dst_row * tensor_cols + j] += value_data[idx];
            }
        }
    } else {
        for i in 0..indices_rows {
            for j in 0..indices_cols {
                let idx = i * indices_cols + j;
                let dst_col = checked_index(indices_data[idx], dim_size);
                result[i * tensor_cols + dst_col] += value_data[idx];
            }
        }
    }
}

/// Select slices from tensor along a dimension using 1D indices.
///
/// Unlike gather, indices is 1D and selects entire slices.
/// For a 2D tensor with dim=0 and indices=[2, 0]:
/// output[0, :] = tensor[2, :]
/// output[1, :] = tensor[0, :]
pub fn select<E: Element + Pod + Default + Copy + Send + Sync>(
    tensor: FlexTensor,
    dim: usize,
    indices: FlexTensor,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let indices = indices.to_contiguous();

    let tensor_shape = tensor.layout().shape();
    let ndims = tensor_shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );
    assert_eq!(
        indices.layout().num_dims(),
        1,
        "select: indices must be 1D, got {} dims",
        indices.layout().num_dims()
    );

    let tensor_data: &[E] = tensor.storage();
    let indices_data = read_indices(&indices);
    let num_indices = indices_data.len();

    // Build output shape: replace dim with num_indices
    let mut output_dims = tensor_shape.to_vec();
    output_dims[dim] = num_indices;
    let output_shape = Shape::from(output_dims);

    // Use optimized 2D implementation with bulk copies
    if ndims == 2 {
        let result = select_2d::<E>(
            tensor_data,
            &indices_data,
            tensor_shape[0],
            tensor_shape[1],
            num_indices,
            dim,
        );
        let bytes = Bytes::from_elems(result);
        return FlexTensor::new(bytes, Layout::contiguous(output_shape), E::dtype());
    }

    // General N-D case
    let tensor_strides: Vec<usize> = compute_strides(tensor_shape);
    let output_strides: Vec<usize> = compute_strides(&output_shape);
    let output_size = output_shape.num_elements();

    // Calculate slice size (elements after dim)
    let slice_size: usize = tensor_strides[dim];

    let select_dim_size = tensor_shape[dim];

    // If dim is the last dimension or we can use bulk copies
    if dim == ndims - 1 || slice_size == 1 {
        // Element-wise with parallelism
        #[cfg(feature = "rayon")]
        let result: Vec<E> = (0..output_size)
            .into_par_iter()
            .map(|out_idx| {
                let mut remaining = out_idx;
                let mut src_idx = 0;
                for d in 0..ndims {
                    let coord = remaining / output_strides[d];
                    remaining %= output_strides[d];
                    if d == dim {
                        let index_val = checked_index(indices_data[coord], select_dim_size);
                        src_idx += index_val * tensor_strides[d];
                    } else {
                        src_idx += coord * tensor_strides[d];
                    }
                }
                tensor_data[src_idx]
            })
            .collect();

        #[cfg(not(feature = "rayon"))]
        #[allow(clippy::needless_range_loop)]
        let result: Vec<E> = {
            let mut result = vec![E::default(); output_size];
            for out_idx in 0..output_size {
                let mut remaining = out_idx;
                let mut src_idx = 0;
                for d in 0..ndims {
                    let coord = remaining / output_strides[d];
                    remaining %= output_strides[d];
                    if d == dim {
                        let index_val = checked_index(indices_data[coord], select_dim_size);
                        src_idx += index_val * tensor_strides[d];
                    } else {
                        src_idx += coord * tensor_strides[d];
                    }
                }
                result[out_idx] = tensor_data[src_idx];
            }
            result
        };

        let bytes = Bytes::from_elems(result);
        return FlexTensor::new(bytes, Layout::contiguous(output_shape), E::dtype());
    }

    // Use bulk copies for contiguous slices
    let mut result = vec![E::default(); output_size];

    // For each position in dimensions before `dim`
    let outer_count = if dim == 0 {
        1
    } else {
        tensor_shape[..dim].iter().product()
    };

    for outer in 0..outer_count {
        let outer_offset_tensor = outer * tensor_strides[if dim == 0 { 0 } else { dim - 1 }];
        let outer_offset_output = outer * output_strides[if dim == 0 { 0 } else { dim - 1 }];

        for (i, &idx) in indices_data.iter().enumerate() {
            let index_val = checked_index(idx, select_dim_size);
            let src_start = outer_offset_tensor + index_val * tensor_strides[dim];
            let dst_start = outer_offset_output + i * output_strides[dim];
            result[dst_start..dst_start + slice_size]
                .copy_from_slice(&tensor_data[src_start..src_start + slice_size]);
        }
    }

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(bytes, Layout::contiguous(output_shape), E::dtype())
}

/// Optimized 2D select with bulk row copies when dim=0.
#[inline]
fn select_2d<E: Element + Pod + Default + Copy + Send + Sync>(
    tensor_data: &[E],
    indices_data: &[isize],
    tensor_rows: usize,
    tensor_cols: usize,
    num_indices: usize,
    dim: usize,
) -> Vec<E> {
    let dim_size = if dim == 0 { tensor_rows } else { tensor_cols };
    let (output_rows, output_cols) = if dim == 0 {
        (num_indices, tensor_cols)
    } else {
        (tensor_rows, num_indices)
    };
    let output_size = output_rows * output_cols;

    // Minimum bytes of output before we consider rayon. Below this, a
    // single-threaded loop is faster because there is not enough work to
    // amortize the work-stealing dispatch overhead.
    #[cfg(feature = "rayon")]
    const PARALLEL_THRESHOLD_BYTES: usize = 4 * 1024 * 1024;

    // Minimum elements per rayon task. Without batching, par_chunks_mut
    // creates one task per row (e.g. 512 single-row tasks of 4 KB each)
    // whose dispatch overhead dominates the actual copy.
    #[cfg(feature = "rayon")]
    const MIN_ELEMS_PER_TASK: usize = 64 * 1024;

    if dim == 0 {
        // SAFETY: the output has exactly num_indices * tensor_cols elements.
        // Both the parallel and serial paths below write every element exactly
        // once via non-overlapping row copies, so no element is left uninitialized.
        let mut result = Vec::with_capacity(output_size);
        #[allow(clippy::uninit_vec)]
        unsafe {
            result.set_len(output_size)
        };

        #[cfg(feature = "rayon")]
        if output_size * size_of::<E>() >= PARALLEL_THRESHOLD_BYTES {
            // Batch multiple rows per rayon task so each task copies at
            // least MIN_ELEMS_PER_TASK elements.
            let rows_per_chunk = (MIN_ELEMS_PER_TASK / tensor_cols).max(1);
            let elems_per_chunk = rows_per_chunk * tensor_cols;
            result.par_chunks_mut(elems_per_chunk).enumerate().for_each(
                |(chunk_idx, dst_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    let chunk_rows = dst_chunk.len() / tensor_cols;
                    for i in 0..chunk_rows {
                        let src_row_idx = checked_index(indices_data[start_row + i], dim_size);
                        let src_start = src_row_idx * tensor_cols;
                        let dst_start = i * tensor_cols;
                        dst_chunk[dst_start..dst_start + tensor_cols]
                            .copy_from_slice(&tensor_data[src_start..src_start + tensor_cols]);
                    }
                },
            );
        } else {
            for (i, &idx) in indices_data.iter().enumerate() {
                let src_row_idx = checked_index(idx, dim_size);
                let src_start = src_row_idx * tensor_cols;
                let dst_start = i * tensor_cols;
                result[dst_start..dst_start + tensor_cols]
                    .copy_from_slice(&tensor_data[src_start..src_start + tensor_cols]);
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            for (i, &idx) in indices_data.iter().enumerate() {
                let src_row_idx = checked_index(idx, dim_size);
                let src_start = src_row_idx * tensor_cols;
                let dst_start = i * tensor_cols;
                result[dst_start..dst_start + tensor_cols]
                    .copy_from_slice(&tensor_data[src_start..src_start + tensor_cols]);
            }
        }

        result
    } else {
        // dim == 1: gather individual elements per row (not contiguous).
        // Zero-init is fine here since the inner loop is per-element anyway.
        let mut result = vec![E::default(); output_size];

        #[cfg(feature = "rayon")]
        if output_size * size_of::<E>() >= PARALLEL_THRESHOLD_BYTES {
            let rows_per_chunk = (MIN_ELEMS_PER_TASK / output_cols).max(1);
            let elems_per_chunk = rows_per_chunk * output_cols;
            result.par_chunks_mut(elems_per_chunk).enumerate().for_each(
                |(chunk_idx, dst_chunk)| {
                    let start_row = chunk_idx * rows_per_chunk;
                    let chunk_rows = dst_chunk.len() / output_cols;
                    for r in 0..chunk_rows {
                        let row = start_row + r;
                        let dst_base = r * output_cols;
                        for (j, &idx) in indices_data.iter().enumerate() {
                            let src_col = checked_index(idx, dim_size);
                            dst_chunk[dst_base + j] = tensor_data[row * tensor_cols + src_col];
                        }
                    }
                },
            );
        } else {
            for row in 0..output_rows {
                for (j, &idx) in indices_data.iter().enumerate() {
                    let src_col = checked_index(idx, dim_size);
                    result[row * output_cols + j] = tensor_data[row * tensor_cols + src_col];
                }
            }
        }

        #[cfg(not(feature = "rayon"))]
        {
            for row in 0..output_rows {
                for (j, &idx) in indices_data.iter().enumerate() {
                    let src_col = checked_index(idx, dim_size);
                    result[row * output_cols + j] = tensor_data[row * tensor_cols + src_col];
                }
            }
        }

        result
    }
}

/// Select add: adds values back to tensor at positions specified by 1D indices.
pub fn select_add<E: Element + Pod + Default + Copy + core::ops::AddAssign + Send + Sync>(
    tensor: FlexTensor,
    dim: usize,
    indices: FlexTensor,
    value: FlexTensor,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let indices = indices.to_contiguous();
    let value = value.to_contiguous();

    let tensor_shape = tensor.layout().shape().clone();
    let value_shape = value.layout().shape();
    let ndims = tensor_shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );
    assert_eq!(
        indices.layout().num_dims(),
        1,
        "select_add: indices must be 1D"
    );

    let tensor_data: &[E] = tensor.storage();
    let indices_data = read_indices(&indices);
    let value_data: &[E] = value.storage();
    let num_indices = indices_data.len();

    // Validate value shape
    for d in 0..ndims {
        if d == dim {
            assert_eq!(
                value_shape[d], num_indices,
                "select_add: value dim {} should be {} (num indices), got {}",
                d, num_indices, value_shape[d]
            );
        } else {
            assert_eq!(
                value_shape[d], tensor_shape[d],
                "select_add: value dim {} should match tensor dim {}, got {}",
                d, tensor_shape[d], value_shape[d]
            );
        }
    }

    let mut result: Vec<E> = tensor_data.to_vec();

    // Use optimized 2D implementation
    if ndims == 2 {
        select_add_2d(
            &mut result,
            &indices_data,
            value_data,
            tensor_shape[0],
            tensor_shape[1],
            num_indices,
            dim,
        );
        let bytes = Bytes::from_elems(result);
        return FlexTensor::new(bytes, Layout::contiguous(tensor_shape), E::dtype());
    }

    // General N-D case
    let tensor_strides: Vec<usize> = compute_strides(&tensor_shape);
    let value_strides: Vec<usize> = compute_strides(value_shape);
    let select_add_dim_size = tensor_shape[dim];

    for (val_idx, &val) in value_data.iter().enumerate() {
        let mut remaining = val_idx;
        let mut dst_idx = 0;
        for d in 0..ndims {
            let coord = remaining / value_strides[d];
            remaining %= value_strides[d];
            if d == dim {
                let index_val = checked_index(indices_data[coord], select_add_dim_size);
                dst_idx += index_val * tensor_strides[d];
            } else {
                dst_idx += coord * tensor_strides[d];
            }
        }
        result[dst_idx] += val;
    }

    let bytes = Bytes::from_elems(result);
    FlexTensor::new(bytes, Layout::contiguous(tensor_shape), E::dtype())
}

/// Optimized 2D select_add.
#[inline]
fn select_add_2d<E: Copy + core::ops::AddAssign>(
    result: &mut [E],
    indices_data: &[isize],
    value_data: &[E],
    tensor_rows: usize,
    tensor_cols: usize,
    num_indices: usize,
    dim: usize,
) {
    let dim_size = if dim == 0 { tensor_rows } else { tensor_cols };
    if dim == 0 {
        for (i, &idx) in indices_data.iter().enumerate() {
            let dst_row = checked_index(idx, dim_size);
            let dst_start = dst_row * tensor_cols;
            let src_start = i * tensor_cols;
            for j in 0..tensor_cols {
                result[dst_start + j] += value_data[src_start + j];
            }
        }
    } else {
        for row in 0..tensor_rows {
            for (j, &idx) in indices_data.iter().enumerate() {
                let dst_col = checked_index(idx, dim_size);
                result[row * tensor_cols + dst_col] += value_data[row * num_indices + j];
            }
        }
    }
}

/// Compute row-major strides for a shape.
#[inline]
fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let ndims = dims.len();
    let mut strides = vec![1usize; ndims];
    for i in (0..ndims.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

// Type-specific wrappers

pub fn gather_f32(tensor: FlexTensor, dim: usize, indices: FlexTensor) -> FlexTensor {
    gather::<f32>(tensor, dim, indices)
}

pub fn gather_f64(tensor: FlexTensor, dim: usize, indices: FlexTensor) -> FlexTensor {
    gather::<f64>(tensor, dim, indices)
}

pub fn gather_i64(tensor: FlexTensor, dim: usize, indices: FlexTensor) -> FlexTensor {
    gather::<i64>(tensor, dim, indices)
}

pub fn scatter_add_f32(
    tensor: FlexTensor,
    dim: usize,
    indices: FlexTensor,
    value: FlexTensor,
) -> FlexTensor {
    scatter_add::<f32>(tensor, dim, indices, value)
}

pub fn scatter_add_f64(
    tensor: FlexTensor,
    dim: usize,
    indices: FlexTensor,
    value: FlexTensor,
) -> FlexTensor {
    scatter_add::<f64>(tensor, dim, indices, value)
}

pub fn scatter_add_i64(
    tensor: FlexTensor,
    dim: usize,
    indices: FlexTensor,
    value: FlexTensor,
) -> FlexTensor {
    scatter_add::<i64>(tensor, dim, indices, value)
}

pub fn select_f32(tensor: FlexTensor, dim: usize, indices: FlexTensor) -> FlexTensor {
    select::<f32>(tensor, dim, indices)
}

pub fn select_f64(tensor: FlexTensor, dim: usize, indices: FlexTensor) -> FlexTensor {
    select::<f64>(tensor, dim, indices)
}

pub fn select_i64(tensor: FlexTensor, dim: usize, indices: FlexTensor) -> FlexTensor {
    select::<i64>(tensor, dim, indices)
}

pub fn select_add_f32(
    tensor: FlexTensor,
    dim: usize,
    indices: FlexTensor,
    value: FlexTensor,
) -> FlexTensor {
    select_add::<f32>(tensor, dim, indices, value)
}

pub fn select_add_f64(
    tensor: FlexTensor,
    dim: usize,
    indices: FlexTensor,
    value: FlexTensor,
) -> FlexTensor {
    select_add::<f64>(tensor, dim, indices, value)
}

pub fn select_add_i64(
    tensor: FlexTensor,
    dim: usize,
    indices: FlexTensor,
    value: FlexTensor,
) -> FlexTensor {
    select_add::<i64>(tensor, dim, indices, value)
}

// Bool-specific operations

pub fn gather_bool(tensor: FlexTensor, dim: usize, indices: FlexTensor) -> FlexTensor {
    gather::<u8>(tensor, dim, indices)
}

/// Scatter OR for bool tensors: ORs values into tensor at indexed positions.
pub fn scatter_or(
    tensor: FlexTensor,
    dim: usize,
    indices: FlexTensor,
    value: FlexTensor,
) -> FlexTensor {
    // Preserve the input tensor's bool dtype for the output.
    let out_dtype = burn_std::BoolDType::from(tensor.dtype());
    let tensor = tensor.to_contiguous();
    let indices = indices.to_contiguous();
    let value = value.to_contiguous();

    let tensor_shape = tensor.layout().shape().clone();
    let indices_shape = indices.layout().shape();
    let value_shape = value.layout().shape();
    let ndims = tensor_shape.num_dims();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );
    assert_eq!(
        indices_shape,
        value_shape,
        "scatter_or: indices shape {:?} must match value shape {:?}",
        indices_shape.to_vec(),
        value_shape.to_vec()
    );

    for i in 0..ndims {
        if i != dim {
            assert_eq!(
                tensor_shape[i], indices_shape[i],
                "scatter_or: shape mismatch at dim {}: tensor {} vs indices {}",
                i, tensor_shape[i], indices_shape[i]
            );
        }
    }

    let tensor_data: &[u8] = tensor.storage();
    let indices_data = read_indices(&indices);
    let value_data: &[u8] = value.storage();

    let mut result: Vec<u8> = tensor_data.to_vec();

    let tensor_strides = compute_strides(&tensor_shape);
    let indices_strides = compute_strides(indices_shape);

    let num_elements = indices_shape.num_elements();

    let scatter_or_dim_size = tensor_shape[dim];

    // Use 2D specialized path
    if ndims == 2 {
        let tensor_cols = tensor_shape[1];
        let indices_rows = indices_shape[0];
        let indices_cols = indices_shape[1];

        if dim == 0 {
            for i in 0..indices_rows {
                for j in 0..indices_cols {
                    let idx = i * indices_cols + j;
                    let dst_row = checked_index(indices_data[idx], scatter_or_dim_size);
                    result[dst_row * tensor_cols + j] |= value_data[idx];
                }
            }
        } else {
            for i in 0..indices_rows {
                for j in 0..indices_cols {
                    let idx = i * indices_cols + j;
                    let dst_col = checked_index(indices_data[idx], scatter_or_dim_size);
                    result[i * tensor_cols + dst_col] |= value_data[idx];
                }
            }
        }
    } else {
        let dim_stride = tensor_strides[dim];
        for idx in 0..num_elements {
            let index_val = checked_index(indices_data[idx], scatter_or_dim_size);
            let dst_idx = compute_gather_index(
                idx,
                index_val,
                dim,
                dim_stride,
                &indices_strides,
                &tensor_strides,
                ndims,
            );
            result[dst_idx] |= value_data[idx];
        }
    }

    crate::ops::comparison::make_bool_tensor(result, tensor_shape, out_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_gather_1d() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], [4]));
        let indices = FlexTensor::from_data(TensorData::new(vec![3i64, 0, 2], [3]));

        let result = gather_f32(tensor, 0, indices);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![40.0, 10.0, 30.0]);
    }

    #[test]
    fn test_gather_2d_dim1() {
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
        ));
        let indices = FlexTensor::from_data(TensorData::new(vec![0i64, 2, 1, 0], [2, 2]));

        let result = gather_f32(tensor, 1, indices);
        assert_eq!(result.layout().shape().to_vec(), vec![2, 2]);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 3.0, 5.0, 4.0]);
    }

    #[test]
    fn test_gather_2d_dim0() {
        // tensor shape [2, 3], gather on dim 0, indices must have cols=3
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
        ));
        // row 0: [1.0, 2.0, 3.0]
        // row 1: [4.0, 5.0, 6.0]
        let indices = FlexTensor::from_data(TensorData::new(vec![1i64, 0, 1, 0, 1, 0], [2, 3]));

        let result = gather_f32(tensor, 0, indices);
        assert_eq!(result.layout().shape().to_vec(), vec![2, 3]);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        // output[i,j] = tensor[indices[i,j], j]
        // [0,0]: tensor[1, 0] = 4.0
        // [0,1]: tensor[0, 1] = 2.0
        // [0,2]: tensor[1, 2] = 6.0
        // [1,0]: tensor[0, 0] = 1.0
        // [1,1]: tensor[1, 1] = 5.0
        // [1,2]: tensor[0, 2] = 3.0
        assert_eq!(data, vec![4.0, 2.0, 6.0, 1.0, 5.0, 3.0]);
    }

    #[test]
    fn test_scatter_add_1d() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![0.0f32, 0.0, 0.0, 0.0], [4]));
        let indices = FlexTensor::from_data(TensorData::new(vec![1i64, 2, 1], [3]));
        let value = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]));

        let result = scatter_add_f32(tensor, 0, indices, value);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![0.0, 4.0, 2.0, 0.0]);
    }

    #[test]
    fn test_scatter_add_2d_dim1() {
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2, 3],
        ));
        let indices = FlexTensor::from_data(TensorData::new(vec![0i64, 2, 1, 0], [2, 2]));
        let value = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));

        let result = scatter_add_f32(tensor, 1, indices, value);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 0.0, 2.0, 4.0, 3.0, 0.0]);
    }

    #[test]
    fn test_select_2d_dim0() {
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
        ));
        let indices = FlexTensor::from_data(TensorData::new(vec![2i64, 0], [2]));

        let result = select_f32(tensor, 0, indices);
        assert_eq!(result.layout().shape().to_vec(), vec![2, 2]);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn test_select_2d_dim1() {
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
        ));
        let indices = FlexTensor::from_data(TensorData::new(vec![2i64, 0], [2]));

        let result = select_f32(tensor, 1, indices);
        assert_eq!(result.layout().shape().to_vec(), vec![2, 2]);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3.0, 1.0, 6.0, 4.0]);
    }

    #[test]
    fn test_select_add_2d_dim0() {
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3, 2],
        ));
        let indices = FlexTensor::from_data(TensorData::new(vec![2i64, 0], [2]));
        let value = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));

        let result = select_add_f32(tensor, 0, indices, value);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3.0, 4.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_gather_i64() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![10i64, 20, 30, 40], [4]));
        let indices = FlexTensor::from_data(TensorData::new(vec![3i64, 0, 2], [3]));

        let result = gather_i64(tensor, 0, indices);
        let data: Vec<i64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![40, 10, 30]);
    }

    #[test]
    fn test_gather_with_i32_indices() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0, 30.0, 40.0], [4]));
        let indices = FlexTensor::from_data(TensorData::new(vec![3i32, 0, 2], [3]));

        let result = gather::<f32>(tensor, 0, indices);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![40.0, 10.0, 30.0]);
    }

    #[test]
    fn test_select_with_i32_indices() {
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
        ));
        let indices = FlexTensor::from_data(TensorData::new(vec![2i32, 0], [2]));

        let result = select::<f32>(tensor, 0, indices);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn test_select_2d_dim0_empty_indices() {
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
        ));
        let indices = FlexTensor::from_data(TensorData::new(Vec::<i64>::new(), [0]));

        let result = select::<f32>(tensor, 0, indices);
        assert_eq!(result.layout().shape().to_vec(), vec![0, 2]);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert!(data.is_empty());
    }

    /// Exercises the uninit buffer path (dim=0) and, when rayon is enabled,
    /// the chunked parallel path (output > 4 MB).
    #[test]
    fn test_select_2d_dim0_large() {
        let rows = 2048;
        let cols = 1024;
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let tensor = FlexTensor::from_data(TensorData::new(data.clone(), [rows, cols]));

        // Select every other row in reverse order.
        let idx: Vec<i64> = (0..rows as i64).rev().step_by(2).collect();
        let num_idx = idx.len();
        let indices = FlexTensor::from_data(TensorData::new(idx.clone(), [num_idx]));

        let result = select::<f32>(tensor, 0, indices);
        assert_eq!(result.layout().shape().to_vec(), vec![num_idx, cols]);
        let out: Vec<f32> = result.into_data().to_vec().unwrap();

        for (i, &row_idx) in idx.iter().enumerate() {
            let expected_start = row_idx as usize * cols;
            let actual = &out[i * cols..(i + 1) * cols];
            let expected = &data[expected_start..expected_start + cols];
            assert_eq!(
                actual, expected,
                "mismatch at output row {i} (src row {row_idx})"
            );
        }
    }
}
