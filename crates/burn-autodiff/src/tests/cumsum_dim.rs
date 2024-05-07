#[burn_tensor_testgen::testgen(ad_cumsum_dim)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn should_diff_cumsum_dim() {
        let device = Default::default();
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]);
        // Original Tensor
        let tensor_0 = TestAutodiffTensor::from_data(data, &device).require_grad();
        // Cumsum Tensor
        let dim = 1;
        let tensor_1 = tensor_0.clone().cumsum_dim(dim);
        // Fake loss
        let loss = tensor_1.clone().sum();
        // Gradients with respect to the original tensor
        let grads = loss.backward();
        // let grads = tensor_1.backward();
        let grad_0 = tensor_0.grad(&grads).unwrap();
        // Gradient is correct
        let grad_0_expected = Data::from([[3., 2., 1.], [3., 2., 1.], [3., 2., 1.]]);
        grad_0.into_data().assert_approx_eq(&grad_0_expected, 2);
    }
}
