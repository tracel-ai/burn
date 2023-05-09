#[burn_tensor_testgen::testgen(ad_maxmin)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn should_diff_max_dim() {
        let tensor_1 = TestADTensor::from_floats([[1.0, 7.0], [-2.0, -3.0]]).require_grad();
        let tensor_2 = TestADTensor::from_floats([[4.0, -7.0], [2.0, 3.0]]).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_1.clone().mul(tensor_3.max_dim(1).unsqueeze());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[50.0, 34.0], [40.0, -10.0]]), 5);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[8.0, 10.0], [56.0, 15.0]]), 5);
    }

    #[test]
    fn should_diff_min_dim() {
        let tensor_1 = TestADTensor::from_floats([[1.0, 7.0], [-2.0, -3.0]]).require_grad();
        let tensor_2 = TestADTensor::from_floats([[4.0, -7.0], [2.0, 3.0]]).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_1.clone().mul(tensor_3.min_dim(1).unsqueeze());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[-42.0, 38.0], [-34.0, -24.0]]), 5);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[10.0, 8.0], [15.0, 56.0]]), 5);
    }
}
