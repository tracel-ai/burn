#[burn_tensor_testgen::testgen(ad_div)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn should_diff_div() {
        let data_1 = Data::from([1.0, 7.0]);
        let data_2 = Data::from([4.0, 7.0]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();

        let tensor_3 = tensor_1.clone().div(tensor_2.clone());
        let grads = tensor_3.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([0.25, 0.1429]), 3);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([-0.0625, -0.1429]), 3);
    }

    #[test]
    fn should_diff_div_scalar() {
        let data = Data::from([1.0, 7.0]);

        let tensor = TestADTensor::from_data(data).require_grad();
        let tensor_out = tensor.clone().div_scalar(4.0);

        let grads = tensor_out.backward();
        let grad = tensor.grad(&grads).unwrap();

        assert_eq!(grad.to_data(), Data::from([0.25, 0.25]));
    }

    #[test]
    fn test_div_complex_1() {
        let data_1: Data<f32, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f32, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f32, 2> = Data::from([[2.0, 2.0], [2.0, 2.0]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();
        let tensor_3 = TestADTensor::from_data(data_3).require_grad();

        let tensor_4 = tensor_1.clone().div(tensor_2.clone());
        let tensor_5 = tensor_4.div(tensor_3);

        let grads = tensor_5.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[0.1250, 0.0714], [0.25, 0.1667]]), 3);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[-0.0312, -0.0714], [-1.6250, 0.1667]]), 3);
    }

    #[test]
    fn test_div_complex_2() {
        let data_1 = Data::from([[0.0, 1.0], [3.0, 4.0]]);
        let data_2 = Data::from([[6.0, 7.0], [9.0, 10.0]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_3.div(tensor_2.clone());

        let grads = tensor_4.backward();
        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[2.00, 2.9286], [1.3667, 2.0]]), 3);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[0.0833, 0.0959], [-0.0556, -0.0671]]), 3);
    }
}
