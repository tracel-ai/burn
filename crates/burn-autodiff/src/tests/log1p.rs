#[burn_tensor_testgen::testgen(ad_log1p)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn should_diff_log1p() {
        let tensor_1 = TestAutodiffTensor::<2>::from([[0.0, 1.0], [3.0, 4.0]]).require_grad();
        let tensor_2 = TestAutodiffTensor::from([[6.0, 7.0], [9.0, 10.0]]).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().log1p());
        let tensor_4 = tensor_3.matmul(tensor_2.clone());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        let expected = TensorData::from([[64.80622, 75.49362], [64.80622, 75.49362]]);
        grad_1.to_data().assert_approx_eq(&expected, 3);

        let expected = TensorData::from([[22.922085, 24.475657], [24.727802, 26.864166]]);
        grad_2.to_data().assert_approx_eq(&expected, 3);
    }
}
