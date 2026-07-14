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

// Tests kept here exercise flex-specific behavior: flip is a zero-copy
// stride-only operation in the flex backend, and the test below verifies
// the underlying buffer pointer is shared across the flip. Correctness
// tests for flip along various axes live in
// crates/burn-backend-tests/tests/tensor/{float,int,bool}/ops/flip.rs and
// run against every backend.
#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

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
