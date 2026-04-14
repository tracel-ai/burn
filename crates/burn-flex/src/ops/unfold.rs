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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_unfold_1d() {
        // Input: [1, 2, 3, 4, 5] shape [5]
        // Unfold dim=0, size=3, step=1
        // Windows: (5 - 3 + 1) / 1 = 3
        // Output shape: [3, 3]
        // Window 0: [1, 2, 3]
        // Window 1: [2, 3, 4]
        // Window 2: [3, 4, 5]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], [5]));
        let result = unfold_f32(tensor, 0, 3, 1);
        assert_eq!(result.layout().shape().to_vec(), vec![3, 3]);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_unfold_1d_step2() {
        // Input: [1, 2, 3, 4, 5, 6] shape [6]
        // Unfold dim=0, size=3, step=2
        // Windows: (6 - 3 + 2) / 2 = 2
        // Output shape: [2, 3]
        // Window 0: [1, 2, 3]
        // Window 1: [3, 4, 5]
        let tensor =
            FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [6]));
        let result = unfold_f32(tensor, 0, 3, 2);
        assert_eq!(result.layout().shape().to_vec(), vec![2, 3]);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_unfold_2d_dim1() {
        // Input: [[1, 2, 3, 4], [5, 6, 7, 8]] shape [2, 4]
        // Unfold dim=1, size=2, step=1
        // Windows: (4 - 2 + 1) / 1 = 3
        // Output shape: [2, 3, 2]
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 4],
        ));
        let result = unfold_f32(tensor, 1, 2, 1);
        assert_eq!(result.layout().shape().to_vec(), vec![2, 3, 2]);
        let data: Vec<f32> = result.into_data().to_vec().unwrap();
        // Row 0: windows [1,2], [2,3], [3,4]
        // Row 1: windows [5,6], [6,7], [7,8]
        assert_eq!(
            data,
            vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0]
        );
    }
}
