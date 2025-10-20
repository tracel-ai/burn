#[burn_tensor_testgen::testgen(ad_cumprod)]
mod tests {
    use super::*;
    use burn_tensor::ops::FloatElem;
    use burn_tensor::{TensorData, Tolerance};

    #[test]
    fn should_diff_cumprod() {
        // Simple test to verify cumprod gradients work
        let device = Default::default();
        let tensor = TestAutodiffTensor::<1>::from_data(TensorData::from([2.0, 3.0, 4.0]), &device)
            .require_grad();

        let output = tensor.clone().cumprod(0);
        let grads = output.sum().backward();
        let grad = tensor.grad(&grads).unwrap();

        // PyTorch reference: [16.0, 10.0, 6.0]
        let expected = TensorData::from([16.0, 10.0, 6.0]);
        grad.to_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&expected, Tolerance::default());
    }

    #[test]
    fn should_diff_cumprod_2d() {
        // Test 2D cumprod gradients
        let device = Default::default();
        let tensor = TestAutodiffTensor::<2>::from_data(
            TensorData::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            &device,
        )
        .require_grad();

        let output = tensor.clone().cumprod(1);
        let grads = output.sum().backward();
        let grad = tensor.grad(&grads).unwrap();

        // PyTorch reference: [[9.0, 4.0, 2.0], [36.0, 28.0, 20.0]]
        let expected = TensorData::from([[9.0, 4.0, 2.0], [36.0, 28.0, 20.0]]);
        grad.to_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&expected, Tolerance::default());
    }

    // TODO: The following tests are currently ignored due to a known limitation
    // in the cumprod gradient implementation. The current implementation uses
    // division (grad / input), which produces NaN when the input contains zeros.
    //
    // A proper fix requires implementing a zero-safe algorithm using exclusive
    // cumulative products (similar to PyTorch's cumprod_backward or JAX's
    // associative_scan approach). This is a non-trivial implementation that
    // requires careful handling of cumulative products in both forward and
    // reverse directions.
    //
    // See: https://github.com/tracel-ai/burn/issues/3864
    //
    // References:
    // - PyTorch: https://github.com/pytorch/pytorch (cumprod_backward)
    // - JAX PR #2596: Parallel prefix scan implementation
    // - TensorFlow Issue #3862: tf.cumprod's gradient produces nans given zeros

    #[test]
    #[ignore = "cumprod gradient with zeros not yet implemented - produces NaN due to division by zero"]
    fn should_diff_cumprod_zero_in_middle() {
        // Test cumprod with zero in the middle - edge case for division
        let device = Default::default();
        let tensor =
            TestAutodiffTensor::<1>::from_data(TensorData::from([2.0, 0.0, 3.0, 4.0]), &device)
                .require_grad();

        let output = tensor.clone().cumprod(0);
        let grads = output.sum().backward();
        let grad = tensor.grad(&grads).unwrap();

        // PyTorch reference: [1.0, 32.0, 0.0, 0.0]
        let expected = TensorData::from([1.0, 32.0, 0.0, 0.0]);
        grad.to_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&expected, Tolerance::default());
    }

    #[test]
    #[ignore = "cumprod gradient with zeros not yet implemented - produces NaN due to division by zero"]
    fn should_diff_cumprod_zero_at_start() {
        // Test cumprod with zero at the beginning
        let device = Default::default();
        let tensor =
            TestAutodiffTensor::<1>::from_data(TensorData::from([0.0, 2.0, 3.0, 4.0]), &device)
                .require_grad();

        let output = tensor.clone().cumprod(0);
        let grads = output.sum().backward();
        let grad = tensor.grad(&grads).unwrap();

        // PyTorch reference: [33.0, 0.0, 0.0, 0.0]
        let expected = TensorData::from([33.0, 0.0, 0.0, 0.0]);
        grad.to_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&expected, Tolerance::default());
    }

    #[test]
    #[ignore = "cumprod gradient with zeros not yet implemented - produces NaN due to division by zero"]
    fn should_diff_cumprod_zero_at_end() {
        // Test cumprod with zero at the end
        let device = Default::default();
        let tensor =
            TestAutodiffTensor::<1>::from_data(TensorData::from([2.0, 3.0, 4.0, 0.0]), &device)
                .require_grad();

        let output = tensor.clone().cumprod(0);
        let grads = output.sum().backward();
        let grad = tensor.grad(&grads).unwrap();

        // PyTorch reference: [16.0, 10.0, 6.0, 24.0]
        let expected = TensorData::from([16.0, 10.0, 6.0, 24.0]);
        grad.to_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&expected, Tolerance::default());
    }

    #[test]
    #[ignore = "cumprod gradient with zeros not yet implemented - produces NaN due to division by zero"]
    fn should_diff_cumprod_multiple_zeros() {
        // Test cumprod with multiple zeros
        let device = Default::default();
        let tensor = TestAutodiffTensor::<1>::from_data(
            TensorData::from([2.0, 0.0, 3.0, 0.0, 5.0]),
            &device,
        )
        .require_grad();

        let output = tensor.clone().cumprod(0);
        let grads = output.sum().backward();
        let grad = tensor.grad(&grads).unwrap();

        // PyTorch reference: [1.0, 8.0, 0.0, 0.0, 0.0]
        let expected = TensorData::from([1.0, 8.0, 0.0, 0.0, 0.0]);
        grad.to_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&expected, Tolerance::default());
    }
}
