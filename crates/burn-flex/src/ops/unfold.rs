//! Unfold operation for sliding window extraction.
//!
//! Implemented as a zero-copy strided view. The output tensor shares storage
//! with the input, using strides to represent overlapping windows.

use alloc::vec::Vec;

use burn_std::Shape;

use crate::{FlexTensor, Layout};

/// Calculate the number of windows that can be extracted from a dimension.
#[inline]
fn calculate_windows(dim_size: usize, window_size: usize, step: usize) -> usize {
    assert!(step > 0, "step must be positive");
    if dim_size + step < window_size {
        0
    } else {
        (dim_size + step - window_size) / step
    }
}

/// Unfold: extract sliding windows from a tensor along a dimension.
///
/// Given a tensor with shape `[pre..., dim_size, post...]`, extracts windows of
/// `size` elements along dimension `dim`, stepping by `step`.
///
/// Returns a tensor with shape `[pre..., windows, post..., size]` where:
/// - `windows = (dim_size - size + step) / step`
/// - The `size` dimension is appended at the end
///
/// This is a zero-copy operation that returns a strided view of the input.
/// The same storage elements may appear in multiple windows (overlapping).
pub fn unfold(tensor: FlexTensor, dim: usize, size: usize, step: usize) -> FlexTensor {
    let input_layout = tensor.layout();
    let shape = input_layout.shape();
    let input_strides = input_layout.strides();
    let start_offset = input_layout.start_offset();
    let ndims = shape.num_dims();
    let dtype = tensor.dtype();

    assert!(
        dim < ndims,
        "dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );
    assert!(size > 0, "window size must be positive");
    assert!(step > 0, "step must be positive");
    assert!(
        shape[dim] >= size,
        "dimension {} has size {} which is smaller than window size {}",
        dim,
        shape[dim],
        size
    );

    let dim_size = shape[dim];
    let windows = calculate_windows(dim_size, size, step);

    // Build output shape: [pre..., windows, post..., size]
    let mut output_dims: Vec<usize> = Vec::with_capacity(ndims + 1);
    for (d, &s) in shape.iter().enumerate() {
        if d == dim {
            output_dims.push(windows);
        } else {
            output_dims.push(s);
        }
    }
    output_dims.push(size); // Append size at the end

    // Build output strides:
    // - Dimensions before `dim`: same stride as input
    // - Dimension `dim` (now windows): input_stride[dim] * step
    // - Dimensions after `dim`: same stride as input
    // - New size dimension (appended): input_stride[dim]
    let mut output_strides: Vec<isize> = Vec::with_capacity(ndims + 1);
    for (d, &s) in input_strides.iter().enumerate() {
        if d == dim {
            // Windows dimension: stride = original_stride * step
            output_strides.push(s * step as isize);
        } else {
            output_strides.push(s);
        }
    }
    // Append stride for the size dimension (position within window)
    output_strides.push(input_strides[dim]);

    let output_shape = Shape::from(output_dims);
    let output_layout = Layout::new(output_shape, output_strides, start_offset);

    // Zero-copy: reuse the same storage with new layout
    FlexTensor::from_arc(tensor.data_arc(), output_layout, dtype)
}

// Type-specific wrappers (all delegate to the generic unfold which is now type-agnostic)

pub fn unfold_f32(tensor: FlexTensor, dim: usize, size: usize, step: usize) -> FlexTensor {
    unfold(tensor, dim, size, step)
}

pub fn unfold_f64(tensor: FlexTensor, dim: usize, size: usize, step: usize) -> FlexTensor {
    unfold(tensor, dim, size, step)
}

pub fn unfold_bool(tensor: FlexTensor, dim: usize, size: usize, step: usize) -> FlexTensor {
    unfold(tensor, dim, size, step)
}

pub fn unfold_int(tensor: FlexTensor, dim: usize, size: usize, step: usize) -> FlexTensor {
    unfold(tensor, dim, size, step)
}

// Correctness of unfold across dtypes and shapes is covered by the
// cross-backend suite in
// crates/burn-backend-tests/tests/tensor/{float,int,bool}/ops/unfold.rs,
// which exercises the flex backend through the public `unfold` op. No
// flex-specific tests remain here.
