#[burn_tensor_testgen::testgen(ad_cummax)]
mod tests {
    use super::*;
    use burn_tensor::ops::FloatElem;
    use burn_tensor::{TensorData, Tolerance};

    #[test]
    fn should_diff_cummax() {
        // Simple test to verify cummax gradients work
        let device = Default::default();
        let tensor = TestAutodiffTensor::<1>::from_data(TensorData::from([1.0, 3.0, 2.0]), &device)
            .require_grad();

        let output = tensor.clone().cummax(0);
        let grads = output.sum().backward();
        let grad = tensor.grad(&grads).unwrap();

        // input:  [1.0, 3.0, 2.0]
        // cummax: [1.0, 3.0, 3.0]
        // Gradient flows to positions where maximum occurred:
        // - output[0] came from input[0] -> grad 1.0
        // - output[1] came from input[1] -> grad 1.0
        // - output[2] came from input[1] -> grad 1.0
        // Expected: [1.0, 2.0, 0.0]
        let expected = TensorData::from([1.0, 2.0, 0.0]);
        grad.to_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&expected, Tolerance::default());
    }

    #[test]
    fn should_diff_cummax_2d() {
        // Test 2D cummax gradients
        let device = Default::default();
        let tensor = TestAutodiffTensor::<2>::from_data(
            TensorData::from([[1.0, 3.0, 2.0], [2.0, 5.0, 4.0]]),
            &device,
        )
        .require_grad();

        let output = tensor.clone().cummax(1);
        let grads = output.sum().backward();
        let grad = tensor.grad(&grads).unwrap();

        // input:  [[1.0, 3.0, 2.0], [2.0, 5.0, 4.0]]
        // cummax: [[1.0, 3.0, 3.0], [2.0, 5.0, 5.0]]
        // Row 0: [1.0, 2.0, 0.0] (position 1 gets grads from positions 1 and 2)
        // Row 1: [1.0, 2.0, 0.0] (position 1 gets grads from positions 1 and 2)
        let expected = TensorData::from([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]]);
        grad.to_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&expected, Tolerance::default());
    }
}
