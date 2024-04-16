#[burn_tensor_testgen::testgen(ad_repeat)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Data};

    #[test]
    fn should_diff_repeat() {
        let data_1 = Data::<f32, 2>::from([[1.0, 7.0], [-2.0, -3.0]]);
        let data_2 = Data::<f32, 2>::from([[4.0], [2.0]]);

        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::from_data(data_1, &device).require_grad();
        let tensor_2 = TestAutodiffTensor::from_data(data_2, &device).require_grad();

        let tensor_3 = tensor_2.clone().repeat(1, 3);

        let tensor_3 = tensor_1.matmul(tensor_3);
        let grads = tensor_3.backward();

        let grad_2 = tensor_2.grad(&grads).unwrap();

        assert_eq!(grad_2.to_data(), Data::from([[-3.0], [12.0]]));
    }
}
