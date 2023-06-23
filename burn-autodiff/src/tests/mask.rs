#[burn_tensor_testgen::testgen(ad_mask)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn should_diff_mask_fill() {
        let data_1 = Data::<f32, 2>::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2 = Data::<f32, 2>::from([[4.0, 7.0], [2.0, 3.0]]);
        let mask = Data::<bool, 2>::from([[true, false], [false, true]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();
        let mask = TestADTensor::from_bool(mask);

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_3.mask_fill(mask, 2.0);
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[7.0, 3.0], [4.0, 2.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[2.0, 1.0], [3.0, 7.0]]));
    }

    #[test]
    fn should_diff_mask_where() {
        let tensor_1 = TestADTensor::from_data([[1.0, 7.0], [2.0, 3.0]]).require_grad();
        let tensor_2 = TestADTensor::from_data([[4.0, 7.0], [2.0, 3.0]]).require_grad();
        let tensor_3 = TestADTensor::from_data([[8.8, 9.8], [10.8, 11.8]]).require_grad();
        let mask = TestADTensor::from_data([[true, false], [false, true]]);

        let tensor_4 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_5 = tensor_4.clone().matmul(tensor_3.clone());
        let tensor_6 = tensor_5.mask_where(mask, tensor_3.clone());
        let grads = tensor_6.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();
        let grad_3 = tensor_3.grad(&grads).unwrap();

        grad_1
            .into_data()
            .assert_approx_eq(&Data::from([[121.8, 55.0], [110.8, 50.0]]), 3);
        grad_2
            .into_data()
            .assert_approx_eq(&Data::from([[27.4, 33.4], [95.0, 115.0]]), 3);
        grad_3
            .into_data()
            .assert_approx_eq(&Data::from([[15., 18.], [23., 29.]]), 3);
    }
}
