#[burn_tensor_testgen::testgen(ad_index_select)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn test_index_select_grad() {
        let tensor_1 =
            TestADTensor::from_data(Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])).require_grad();
        let indexes = TestADTensor::from_data(Data::from([[2, 1, 0, 1, 2], [1, 0, 2, 1, 0]]));

        let tensor_2 = tensor_1.clone().matmul(tensor_1.clone().transpose());
        let tensor_3 = tensor_1.clone().index_select(indexes);
        let tensor_4 = tensor_2.matmul(tensor_3);

        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();

        assert_eq!(
            grad_1.into_data(),
            Data::from([[94., 150., 187.], [242., 305., 304.]])
        );
    }

    #[test]
    fn test_index_select_assign_grad() {
        let tensor_1 =
            TestADTensor::from_data(Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])).require_grad();
        let values =
            TestADTensor::from_data(Data::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).require_grad();
        let indexes = TestADTensor::from_data(Data::from([[2, 1, 0], [2, 0, 1]]));

        let tensor_2 = tensor_1.clone().matmul(tensor_1.clone().transpose());
        let tensor_3 = tensor_1
            .clone()
            .index_select_assign(indexes, values.clone());
        let tensor_4 = tensor_2.matmul(tensor_3);

        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = values.grad(&grads).unwrap();

        assert_eq!(
            grad_1.into_data(),
            Data::from([[127., 181., 235.], [226., 316., 406.]])
        );
        assert_eq!(
            grad_2.into_data(),
            Data::from([[19., 19., 19.], [64., 64., 64.]])
        );
    }
}
