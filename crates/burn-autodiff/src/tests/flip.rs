#[burn_tensor_testgen::testgen(ad_flip)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn should_diff_flip() {
        let data_1: Data<f32, 3> = Data::from([[[1.0, 7.0], [2.0, 3.0]]]); // 1x2x2
        let data_2: Data<f32, 3> = Data::from([[[3.0, 2.0, 7.0], [3.0, 3.2, 1.0]]]); // 1x2x3

        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::from_data(data_1, &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

        let tensor_3 = tensor_2.clone().flip([1, 2]);
        let tensor_4 = tensor_1.clone().matmul(tensor_3);
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_1.to_data(), Data::from([[[7.2, 12.0], [7.2, 12.0]]])); // 1x2x2
        assert_eq!(
            grad_2.to_data(),
            Data::from([[[10.0, 10.0, 10.0], [3.0, 3.0, 3.0]]]) // 1x2x3
        );
    }
}
