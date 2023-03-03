#[burn_tensor_testgen::testgen(ad_mul)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn should_diff_mul() {
        let data_1 = Data::from([1.0, 7.0]);
        let data_2 = Data::from([4.0, 7.0]);

        let tensor_1 = TestADTensor::from_data(data_1.clone()).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2.clone()).require_grad();

        let tensor_3 = tensor_1.clone().mul(tensor_2.clone());
        let grads = tensor_3.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), data_2);
        assert_eq!(grad_2.to_data(), data_1);
        assert_eq!(tensor_3.into_data(), Data::from([4.0, 49.0]));
    }

    #[test]
    fn should_diff_mul_scalar() {
        let data = Data::from([2.0, 5.0]);

        let tensor = TestADTensor::from_data(data).require_grad();
        let tensor_out = tensor.clone().mul_scalar(4.0);

        let grads = tensor_out.backward();
        let grad = tensor.grad(&grads).unwrap();

        assert_eq!(tensor_out.into_data(), Data::from([8.0, 20.0]));
        assert_eq!(grad.to_data(), Data::from([4.0, 4.0]));
    }

    #[test]
    fn test_mul_complex_1() {
        let data_1: Data<f32, 2> = Data::from([[1.0, 7.0], [13.0, -3.0]]);
        let data_2: Data<f32, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f32, 2> = Data::from([[2.0, 2.0], [2.0, 2.0]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();
        let tensor_3 = TestADTensor::from_data(data_3).require_grad();

        let tensor_4 = tensor_1.clone().mul(tensor_2.clone());
        let tensor_5 = tensor_4.mul(tensor_3);
        let tensor_6 = tensor_1.clone().mul(tensor_5);

        let grads = tensor_6.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(
            grad_1.to_data(),
            Data::from([[16.0, 196.0], [104.0, -36.0]])
        );
        assert_eq!(grad_2.to_data(), Data::from([[2.0, 98.0], [338.0, 18.0]]));
    }
}
