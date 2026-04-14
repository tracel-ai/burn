//! Flip operation for reversing tensor elements along axes.
//!
//! With signed strides, flip is a zero-copy operation that simply negates
//! the stride and adjusts the start offset for each flipped axis.

use crate::FlexTensor;

/// Flip tensor elements along specified axes.
///
/// This is a zero-copy operation using negative strides.
pub fn flip(tensor: FlexTensor, axes: &[usize]) -> FlexTensor {
    if axes.is_empty() {
        return tensor;
    }

    let new_layout = tensor.layout().flip(axes);
    tensor.with_layout(new_layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[test]
    fn test_flip_1d() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let flipped = flip(tensor, &[0]);
        let data: Vec<f32> = flipped.into_data().to_vec().unwrap();
        assert_eq!(data, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_flip_2d_axis0() {
        // [[1, 2], [3, 4]] -> [[3, 4], [1, 2]]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));
        let flipped = flip(tensor, &[0]);
        let data: Vec<f32> = flipped.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_flip_2d_axis1() {
        // [[1, 2], [3, 4]] -> [[2, 1], [4, 3]]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));
        let flipped = flip(tensor, &[1]);
        let data: Vec<f32> = flipped.into_data().to_vec().unwrap();
        assert_eq!(data, vec![2.0, 1.0, 4.0, 3.0]);
    }

    #[test]
    fn test_flip_2d_both_axes() {
        // [[1, 2], [3, 4]] -> [[4, 3], [2, 1]]
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));
        let flipped = flip(tensor, &[0, 1]);
        let data: Vec<f32> = flipped.into_data().to_vec().unwrap();
        assert_eq!(data, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_flip_empty_axes() {
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [3]));
        let flipped = flip(tensor, &[]);
        let data: Vec<f32> = flipped.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_flip_is_zero_copy() {
        // Verify flip doesn't copy data by checking it shares the same underlying storage
        let tensor = FlexTensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]));
        let tensor_ptr = tensor.bytes().as_ptr();
        let flipped = flip(tensor, &[0]);
        let flipped_ptr = flipped.bytes().as_ptr();
        assert_eq!(
            tensor_ptr, flipped_ptr,
            "flip should share underlying storage"
        );
    }
}
