#[burn_tensor_testgen::testgen(ad_mask)]
mod tests {
    use super::*;
    use burn_tensor::{BoolTensor, Data};

    #[test]
    fn should_diff_mask() {
        let data_1 = Data::<f32, 2>::from([[1.0, 7.0], [2.0, 3.0]]);
        let data_2 = Data::<f32, 2>::from([[4.0, 7.0], [2.0, 3.0]]);
        let mask = Data::<bool, 2>::from([[true, false], [false, true]]);

        let tensor_1 = TestADTensor::from_data(data_1);
        let tensor_2 = TestADTensor::from_data(data_2);
        let mask = BoolTensor::from_data(mask);

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = tensor_3.mask_fill(mask, 2.0);
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[7.0, 3.0], [4.0, 2.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[2.0, 1.0], [3.0, 7.0]]));
    }
}
