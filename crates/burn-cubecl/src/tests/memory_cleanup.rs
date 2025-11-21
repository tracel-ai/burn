#[burn_tensor_testgen::testgen(memory_cleanup)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn test_memory_cleanup_after_linear() {
        let device = Default::default();

        // Create a tensor with require_grad
        let input = TestAutodiffTensor::<2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        )
        .require_grad();

        // Create a Param tensor
        let weight = TestAutodiffTensor::<2>::from_data(
            TensorData::from([[0.5, 0.5], [0.5, 0.5]]),
            &device,
        );

        // Linear model test: simple matmul
        let output = input.clone().matmul(weight);

        // Call memory_cleanup
        TestAutodiffBackend::memory_cleanup(&device);

        // If it doesn't panic, the test passes
    }
}