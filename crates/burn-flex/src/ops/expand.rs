//! Expand operation for broadcasting tensors to larger shapes.

use alloc::vec;
use alloc::vec::Vec;
use burn_std::Shape;

use crate::{FlexTensor, Layout};

/// Compute the broadcast shape of two tensors.
///
/// Returns the shape that both tensors can be expanded to for element-wise operations.
pub fn broadcast_shape(lhs: &Shape, rhs: &Shape) -> Shape {
    let max_dims = lhs.num_dims().max(rhs.num_dims());
    let mut result = vec![0; max_dims];

    for (i, out) in result.iter_mut().enumerate() {
        let lhs_idx = i as isize + lhs.num_dims() as isize - max_dims as isize;
        let rhs_idx = i as isize + rhs.num_dims() as isize - max_dims as isize;

        let lhs_dim = if lhs_idx >= 0 {
            lhs[lhs_idx as usize]
        } else {
            1
        };
        let rhs_dim = if rhs_idx >= 0 {
            rhs[rhs_idx as usize]
        } else {
            1
        };

        if lhs_dim == rhs_dim {
            *out = lhs_dim;
        } else if lhs_dim == 1 {
            *out = rhs_dim;
        } else if rhs_dim == 1 {
            *out = lhs_dim;
        } else {
            panic!(
                "broadcast_shape: incompatible dimensions {} and {} at position {}",
                lhs_dim, rhs_dim, i
            );
        }
    }

    Shape::from(result)
}

/// Broadcast two tensors to the same shape for binary operations.
pub fn broadcast_binary(lhs: FlexTensor, rhs: FlexTensor) -> (FlexTensor, FlexTensor) {
    let lhs_shape = lhs.layout().shape().clone();
    let rhs_shape = rhs.layout().shape().clone();

    if lhs_shape == rhs_shape {
        return (lhs, rhs);
    }

    let target = broadcast_shape(&lhs_shape, &rhs_shape);

    let lhs_expanded = if lhs_shape == target {
        lhs
    } else {
        expand(lhs, target.clone())
    };
    let rhs_expanded = if rhs_shape == target {
        rhs
    } else {
        expand(rhs, target)
    };

    (lhs_expanded, rhs_expanded)
}

/// Expand a tensor to a larger shape by broadcasting.
///
/// Dimensions of size 1 can be expanded to any size. The result is a view
/// that doesn't copy data - it uses stride 0 for expanded dimensions.
pub fn expand(tensor: FlexTensor, target_shape: Shape) -> FlexTensor {
    // Capture values we need before consuming tensor
    let src_dims = tensor.layout().shape().to_vec();
    let src_strides = tensor.layout().strides().to_vec();
    let start_offset = tensor.layout().start_offset();
    let dtype = tensor.dtype();

    let src_ndims = src_dims.len();
    let target_ndims = target_shape.num_dims();

    // Prepend 1s to source shape if needed (for broadcasting like [3] -> [2, 3])
    let dim_diff = target_ndims.saturating_sub(src_ndims);

    let mut new_strides = Vec::with_capacity(target_ndims);

    for i in 0..target_ndims {
        let target_dim = target_shape[i];

        if i < dim_diff {
            // New dimension prepended - must be broadcastable from size 1
            new_strides.push(0);
        } else {
            let src_idx = i - dim_diff;
            let src_dim = src_dims[src_idx];
            let src_stride = src_strides[src_idx];

            if src_dim == target_dim {
                // Same size - keep stride
                new_strides.push(src_stride);
            } else if src_dim == 1 {
                // Broadcast dimension - stride becomes 0
                new_strides.push(0);
            } else {
                panic!(
                    "expand: cannot expand dimension {} from {} to {}",
                    i, src_dim, target_dim
                );
            }
        }
    }

    let new_layout = Layout::new(target_shape, new_strides, start_offset);
    FlexTensor::from_arc(tensor.data_arc(), new_layout, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_expand_1d_to_2d() {
        // [3] -> [2, 3]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]));
        let expanded = expand(tensor, Shape::new([2, 3]));

        assert_eq!(expanded.layout().shape().to_vec(), vec![2, 3]);
        assert_eq!(expanded.layout().strides(), &[0, 1]);
    }

    #[test]
    fn test_expand_broadcast_dim() {
        // [3, 1] -> [3, 4]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3, 1]));
        let expanded = expand(tensor, Shape::new([3, 4]));

        assert_eq!(expanded.layout().shape().to_vec(), vec![3, 4]);
        // Original strides for [3, 1] would be [1, 1]
        // After expand to [3, 4], stride for dim 1 becomes 0
        assert_eq!(expanded.layout().strides()[1], 0);
    }

    #[test]
    fn test_expand_same_shape() {
        // [2, 3] -> [2, 3] (no change)
        let tensor = FlexTensor::from_data(TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
        ));
        let original_strides = tensor.layout().strides().to_vec();
        let expanded = expand(tensor, Shape::new([2, 3]));

        assert_eq!(expanded.layout().shape().to_vec(), vec![2, 3]);
        assert_eq!(expanded.layout().strides(), &original_strides);
    }

    // === Non-contiguous tensor tests ===

    #[test]
    fn test_expand_transposed() {
        // [[1, 2], [3, 4]] transposed -> [[1, 3], [2, 4]] with strides [1, 2]
        // Expand [2, 2] -> [3, 2, 2] by prepending dimension
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));
        let transposed = tensor.transpose(0, 1);
        assert!(!transposed.is_contiguous());
        assert_eq!(transposed.layout().strides(), &[1, 2]);

        let expanded = expand(transposed, Shape::new([3, 2, 2]));
        assert_eq!(expanded.layout().shape().to_vec(), vec![3, 2, 2]);
        // New dim with stride 0, original strides preserved
        assert_eq!(expanded.layout().strides(), &[0, 1, 2]);

        // Verify content: should see same transposed values repeated 3 times
        let data: Vec<f32> = expanded.into_data().to_vec().unwrap();
        // [[1, 3], [2, 4]] repeated 3 times
        assert_eq!(
            data,
            vec![1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0, 1.0, 3.0, 2.0, 4.0]
        );
    }

    #[test]
    fn test_expand_flipped_1d() {
        // [1, 2, 3] flipped -> [3, 2, 1] with negative stride
        // Expand [3] -> [2, 3]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]));
        let flipped = crate::ops::flip::flip(tensor, &[0]);
        assert!(flipped.layout().strides()[0] < 0);

        let expanded = expand(flipped, Shape::new([2, 3]));
        assert_eq!(expanded.layout().shape().to_vec(), vec![2, 3]);
        // Stride 0 for new broadcast dim, negative stride preserved
        assert_eq!(expanded.layout().strides()[0], 0);
        assert!(expanded.layout().strides()[1] < 0);

        // Verify content: [3, 2, 1] repeated twice
        let data: Vec<f32> = expanded.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3.0, 2.0, 1.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_expand_flipped_2d_preserves_negative_stride() {
        // [[1, 2], [3, 4]] with axis 0 flipped -> [[3, 4], [1, 2]]
        // Shape [2, 2] with strides [-2, 1] (negative on axis 0)
        // Expand [2, 2] -> [3, 2, 2]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));
        let flipped = crate::ops::flip::flip(tensor, &[0]);
        assert!(flipped.layout().strides()[0] < 0);
        assert_eq!(flipped.layout().strides()[1], 1);

        let expanded = expand(flipped, Shape::new([3, 2, 2]));
        assert_eq!(expanded.layout().shape().to_vec(), vec![3, 2, 2]);
        // Stride 0 for broadcast, negative stride preserved for axis 1, positive for axis 2
        assert_eq!(expanded.layout().strides()[0], 0);
        assert!(expanded.layout().strides()[1] < 0);
        assert_eq!(expanded.layout().strides()[2], 1);

        // Verify content
        let data: Vec<f32> = expanded.into_data().to_vec().unwrap();
        // [[3, 4], [1, 2]] repeated 3 times
        assert_eq!(
            data,
            vec![3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0]
        );
    }

    #[test]
    fn test_expand_narrowed_preserves_offset() {
        // [0, 1, 2, 3, 4] narrowed to [1, 2, 3] with offset 1
        // Expand [3] -> [2, 3]
        let tensor = FlexTensor::from_data(TensorData::new(vec![0.0f32, 1.0, 2.0, 3.0, 4.0], [5]));
        let narrowed = tensor.narrow(0, 1, 3);
        assert_eq!(narrowed.layout().start_offset(), 1);

        let expanded = expand(narrowed, Shape::new([2, 3]));
        assert_eq!(expanded.layout().shape().to_vec(), vec![2, 3]);
        // Start offset preserved
        assert_eq!(expanded.layout().start_offset(), 1);

        // Verify content: [1, 2, 3] repeated twice
        let data: Vec<f32> = expanded.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_broadcast_binary_with_flipped() {
        // One tensor flipped, broadcast to same shape
        // lhs: [1, 2, 3, 4] flipped -> [4, 3, 2, 1], shape [4]
        // rhs: [[1], [2]], shape [2, 1] -> broadcast to [2, 4]
        // After broadcast: lhs [2, 4], rhs [2, 4]
        let lhs = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let lhs = crate::ops::flip::flip(lhs, &[0]);
        assert!(lhs.layout().strides()[0] < 0);

        let rhs = FlexTensor::from_data(TensorData::new(vec![10.0f32, 20.0], [2, 1]));

        let (lhs_bc, rhs_bc) = broadcast_binary(lhs, rhs);
        assert_eq!(lhs_bc.layout().shape().to_vec(), vec![2, 4]);
        assert_eq!(rhs_bc.layout().shape().to_vec(), vec![2, 4]);

        // lhs should have stride 0 in dim 0 (broadcast), negative stride in dim 1 (from flip)
        assert_eq!(lhs_bc.layout().strides()[0], 0);
        assert!(lhs_bc.layout().strides()[1] < 0);

        // Verify lhs content: [4, 3, 2, 1] repeated twice
        let lhs_data: Vec<f32> = lhs_bc.into_data().to_vec().unwrap();
        assert_eq!(lhs_data, vec![4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0]);
    }
}
