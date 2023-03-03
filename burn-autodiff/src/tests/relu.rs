#[burn_tensor_testgen::testgen(ad_relu)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Data};

    #[test]
    fn should_diff_relu() {
        let data_1 = Data::<f32, 2>::from([[1.0, 7.0], [-2.0, -3.0]]);
        let data_2 = Data::<f32, 2>::from([[4.0, -7.0], [2.0, 3.0]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone());
        let tensor_4 = activation::relu(tensor_3);
        let tensor_5 = tensor_4.matmul(tensor_2.clone());
        let grads = tensor_5.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[-47.0, 9.0], [-35.0, 15.0]]));
        assert_eq!(grad_2.to_data(), Data::from([[15.0, 13.0], [-2.0, 39.0]]));
    }
}
