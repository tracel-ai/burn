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
}
