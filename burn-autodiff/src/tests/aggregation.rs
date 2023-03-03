#[burn_tensor_testgen::testgen(ad_aggregation)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn should_diff_mean() {
        let data_1 = Data::<f32, 2>::from([[1.0, 7.0], [-2.0, -3.0]]);
        let data_2 = Data::<f32, 2>::from([[4.0, -7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_1.clone().mul(tensor_3.mean().unsqueeze());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[3.5, 9.5], [3.5, 9.5]]), 5);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[-0.75, -0.75], [3.0, 3.0]]), 5);
    }

    #[test]
    fn should_diff_sum_1() {
        let data_1 = Data::<f32, 2>::from([[1.0, 7.0], [-2.0, -3.0]]);
        let data_2 = Data::<f32, 2>::from([[4.0, -7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_1.clone().mul(tensor_3.sum().unsqueeze());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[14.0, 38.0], [14.0, 38.0]]), 5);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[-3.0, -3.0], [12.0, 12.0]]), 5);
    }

    #[test]
    fn should_diff_sum_2() {
        let data_1 = Data::from([[0.0, 1.0], [3.0, 4.0]]);
        let data_2 = Data::from([[6.0, 7.0], [9.0, 10.0]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_3.clone().sum_dim(1);
        let tensor_5 = tensor_4.mul(tensor_3);

        let grads = tensor_5.sum().backward();
        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[494.0, 722.0], [2990.0, 4370.0]]), 3);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[690.0, 690.0], [958.0, 958.0]]), 3);
    }

    #[test]
    fn should_diff_mean_dim() {
        let data_1 = Data::<f32, 2>::from([[1.0, 7.0], [-2.0, -3.0]]);
        let data_2 = Data::<f32, 2>::from([[4.0, -7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_1.clone().mul(tensor_3.mean_dim(1).unsqueeze());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[4.0, 36.0], [3.0, -17.0]]), 5);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[9.0, 9.0], [35.5, 35.5]]), 5);
    }

    #[test]
    fn should_diff_sum_dim() {
        let data_1 = Data::<f32, 2>::from([[1.0, 7.0], [-2.0, -3.0]]);
        let data_2 = Data::<f32, 2>::from([[4.0, -7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_1.clone().mul(tensor_3.sum_dim(1).unsqueeze());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[8.0, 72.0], [6.0, -34.0]]), 5);
        grad_2
            .to_data()
            .assert_approx_eq(&Data::from([[18.0, 18.0], [71.0, 71.0]]), 5);
    }
}
