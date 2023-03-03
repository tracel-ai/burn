#[burn_tensor_testgen::testgen(ad_sub)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn should_diff_sub() {
        let data_1 = Data::from([2.0, 5.0]);
        let data_2 = Data::from([4.0, 1.0]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();

        let tensor_3 = tensor_1.clone().sub(tensor_2.clone());
        let grads = tensor_3.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([1.0, 1.0]));
        assert_eq!(grad_2.to_data(), Data::from([-1.0, -1.0]));
        assert_eq!(tensor_3.into_data(), Data::from([-2.0, 4.0]));
    }

    #[test]
    fn should_diff_sub_scalar() {
        let data = Data::from([2.0, 10.0]);
        let tensor = TestADTensor::from_data(data).require_grad();
        let tensor_out = tensor.clone().sub_scalar(5.0);
        let grads = tensor_out.backward();

        let grad = tensor.grad(&grads).unwrap();

        assert_eq!(grad.to_data(), Data::from([1.0, 1.0]));
        assert_eq!(tensor_out.into_data(), Data::from([-3.0, 5.0]));
    }

    #[test]
    fn test_sub_complex_1() {
        let data_1: Data<f32, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f32, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f32, 2> = Data::from([[2.0, 2.0], [2.0, 2.0]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();
        let tensor_3 = TestADTensor::from_data(data_3).require_grad();

        let tensor_4 = tensor_1.clone().sub(tensor_2.clone());
        let tensor_5 = tensor_4.sub(tensor_3).sub_scalar(5.0);
        let tensor_6 = tensor_1.clone().sub(tensor_5);

        let grads = tensor_6.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[0.0, 0.0], [0.0, 0.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[1.0, 1.0], [1.0, 1.0]]));
    }
}
