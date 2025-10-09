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

        // PyTorch reference: [1.0, 2.0, 0.0]
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

        // PyTorch reference: [[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]]
        let expected = TensorData::from([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]]);
        grad.to_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&expected, Tolerance::default());
    }

    #[test]
    fn should_diff_cummax_duplicate_values() {
        // Test with duplicate maximum values - critical edge case
        let device = Default::default();
        let tensor =
            TestAutodiffTensor::<1>::from_data(TensorData::from([1.0, 3.0, 3.0, 2.0]), &device)
                .require_grad();

        let output = tensor.clone().cummax(0);
        let grads = output.sum().backward();
        let grad = tensor.grad(&grads).unwrap();

        // input:  [1.0, 3.0, 3.0, 2.0]
        // cummax: [1.0, 3.0, 3.0, 3.0]
        // PyTorch reference: [1.0, 1.0, 2.0, 0.0]
        // Position 2 gets grad from itself + position 3
        let expected = TensorData::from([1.0, 1.0, 2.0, 0.0]);
        grad.to_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&expected, Tolerance::default());
    }

    #[test]
    fn should_diff_cummax_all_same() {
        // Test with all same values
        let device = Default::default();
        let tensor = TestAutodiffTensor::<1>::from_data(TensorData::from([2.0, 2.0, 2.0]), &device)
            .require_grad();

        let output = tensor.clone().cummax(0);
        let grads = output.sum().backward();
        let grad = tensor.grad(&grads).unwrap();

        // PyTorch reference: [1.0, 1.0, 1.0]
        // Each position matches cummax, so each gets its own gradient
        let expected = TensorData::from([1.0, 1.0, 1.0]);
        grad.to_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&expected, Tolerance::default());
    }

    #[test]
    fn should_diff_cummax_increasing() {
        // Test with increasing sequence
        let device = Default::default();
        let tensor =
            TestAutodiffTensor::<1>::from_data(TensorData::from([1.0, 2.0, 3.0, 4.0]), &device)
                .require_grad();

        let output = tensor.clone().cummax(0);
        let grads = output.sum().backward();
        let grad = tensor.grad(&grads).unwrap();

        // PyTorch reference: [1.0, 1.0, 1.0, 1.0]
        // Each position is a new maximum
        let expected = TensorData::from([1.0, 1.0, 1.0, 1.0]);
        grad.to_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&expected, Tolerance::default());
    }

    #[test]
    fn should_diff_cummax_2d_duplicates() {
        // Test 2D with duplicate values
        let device = Default::default();
        let tensor = TestAutodiffTensor::<2>::from_data(
            TensorData::from([[1.0, 3.0, 3.0, 2.0], [2.0, 5.0, 5.0, 4.0]]),
            &device,
        )
        .require_grad();

        let output = tensor.clone().cummax(1);
        let grads = output.sum().backward();
        let grad = tensor.grad(&grads).unwrap();

        // PyTorch reference: [[1.0, 1.0, 2.0, 0.0], [1.0, 1.0, 2.0, 0.0]]
        let expected = TensorData::from([[1.0, 1.0, 2.0, 0.0], [1.0, 1.0, 2.0, 0.0]]);
        grad.to_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&expected, Tolerance::default());
    }
}
