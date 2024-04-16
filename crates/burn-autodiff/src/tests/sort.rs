#[burn_tensor_testgen::testgen(ad_sort)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn should_diff_sort() {
        let device = Default::default();
        let tensor_1 =
            TestAutodiffTensor::from_floats([[1.0, 7.0], [-2.0, -3.0]], &device).require_grad();
        let tensor_2 =
            TestAutodiffTensor::from_floats([[4.0, -7.0], [2.0, 3.0]], &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_1.clone().mul(tensor_3.sort(1));
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[35.0, 35.0], [-1.0, -8.0]]), 5);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[11.0, 7.0], [55.0, 16.0]]), 5);
    }

    #[test]
    fn should_diff_sort_with_indices() {
        let device = Default::default();
        let tensor_1 =
            TestAutodiffTensor::from_floats([[1.0, 7.0], [-2.0, -3.0]], &device).require_grad();
        let tensor_2 =
            TestAutodiffTensor::from_floats([[4.0, -7.0], [2.0, 3.0]], &device).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_1.clone().mul(tensor_3.sort_with_indices(1).0);
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[35.0, 35.0], [-1.0, -8.0]]), 5);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[11.0, 7.0], [55.0, 16.0]]), 5);
    }
}
