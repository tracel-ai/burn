#[burn_tensor_testgen::testgen(ad_abs)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn should_diff_abs() {
        let data_1 = TensorData::from([[0.0, -1.0], [3.0, 4.0]]);
        let data_2 = TensorData::from([[6.0, 7.0], [9.0, -10.0]]);

        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().abs());
        let tensor_4 = tensor_3.matmul(tensor_2.clone());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        let expected = TensorData::from([[71.0, 107.0], [71.0, 107.0]]);
        grad_1.to_data().assert_approx_eq(&expected, 3);

        let expected = TensorData::from([[84.0, 42.0], [90.0, 54.0]]);
        grad_2.to_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn should_diff_abs_no_nans() {
        let data_1 = TensorData::from([[6.0, 7.0], [9.0, -10.0]]);
        let data_2 = TensorData::from([[0.0, -1.0], [3.0, 4.0]]);

        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::<2>::from_data(data_1, &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().abs());
        let grads = tensor_3.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        let expected = TensorData::from([[1.0, 7.0], [1.0, 7.0]]);
        grad_1.to_data().assert_approx_eq(&expected, 3);

        let expected = TensorData::from([[0.0, -15.0], [-3.0, -3.0]]);
        grad_2.to_data().assert_approx_eq(&expected, 3);

        let contains_nan = grad_2.contains_nan();
        assert_eq!(contains_nan.into_scalar(), false);
    }
}
