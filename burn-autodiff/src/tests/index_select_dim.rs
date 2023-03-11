#[burn_tensor_testgen::testgen(ad_index_select_dim)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn test_select_grad() {
        let tensor_1 =
            TestADTensor::from_data(Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])).require_grad();
        let values =
            TestADTensor::from_data(Data::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).require_grad();
        let indexes = TestADTensor::from_data(Data::from([1, 0]));

        let tensor_2 = tensor_1.clone().matmul(tensor_1.clone().transpose());
        let tensor_3 = tensor_1
            .clone()
            .index_select_dim_assign(0, indexes, values.clone());
        let tensor_4 = tensor_2.matmul(tensor_3);

        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = values.grad(&grads).unwrap();

        assert_eq!(
            grad_1.into_data(),
            Data::from([[127., 199., 271.], [172., 244., 316.]])
        );
        assert_eq!(
            grad_2.into_data(),
            Data::from([[64., 64., 64.], [19., 19., 19.]])
        );
    }
}
