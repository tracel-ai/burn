#[burn_tensor_testgen::testgen(ad_slice)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn should_diff_matmul_with_slice() {
        let data_1: Data<f32, 2> = Data::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2: Data<f32, 2> = Data::from([[4.0, 7.0, 100.0], [2.0, 3.0, 15.0]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();

        let tensor_3 = tensor_2.clone().slice([0..2, 0..2]);
        let tensor_4 = tensor_1.clone().matmul(tensor_3);
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[11.0, 5.0], [11.0, 5.0]]));
        assert_eq!(
            grad_2.to_data(),
            Data::from([[3.0, 3.0, 0.0], [10.0, 10.0, 0.0]])
        );
    }

    #[test]
    fn should_diff_matmul_with_slice_assign() {
        let data_1: Data<f32, 2> = Data::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2: Data<f32, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_assigned: Data<f32, 2> = Data::from([[9.0]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();
        let tensor_assigned = TestADTensor::from_data(data_assigned).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_3.slice_assign([0..1, 0..1], tensor_assigned);
        let tensor_5 = tensor_4.matmul(tensor_1.clone());

        let grads = tensor_5.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[58.0, 38.0], [118.0, 82.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[16.0, 15.0], [24.0, 50.0]]));
    }

    #[test]
    fn should_diff_matmul_with_slice_assign_complex() {
        let data_1: Data<f32, 2> = Data::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2: Data<f32, 2> = Data::from([[4.0, 7.0], [2.0, 3.0]]);
        let data_3: Data<f32, 2> = Data::from([[9.0]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();
        let tensor_3 = TestADTensor::from_data(data_3).require_grad();

        let tensor_4 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_5 = tensor_2.clone().slice([0..1, 0..1]);
        let tensor_6 = tensor_5.mul(tensor_3.clone());
        let tensor_7 = tensor_4.slice_assign([0..1, 0..1], tensor_6);
        let tensor_8 = tensor_7.matmul(tensor_1.clone());

        let grads = tensor_8.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();
        let grad_3 = tensor_3.grad(&grads).unwrap();

        assert_eq!(grad_3.to_data(), Data::from([[32.0]]));
        assert_eq!(grad_1.to_data(), Data::from([[85.0, 65.0], [118.0, 82.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[88.0, 15.0], [24.0, 50.0]]));
    }
}
