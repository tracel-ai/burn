#[burn_tensor_testgen::testgen(ad_cos)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn should_diff_cos() {
        let data_1 = Data::<f32, 2>::from([[0.0, 1.0], [3.0, 4.0]]);
        let data_2 = Data::<f32, 2>::from([[6.0, 7.0], [9.0, 10.0]]);

        let tensor_1 = TestADTensor::from_data(data_1).require_grad();
        let tensor_2 = TestADTensor::from_data(data_2).require_grad();

        let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().cos());
        let tensor_4 = tensor_3.matmul(tensor_2.clone());
        let grads = tensor_4.backward();

        let grad_1 = tensor_1.grad(&grads).unwrap();
        let grad_2 = tensor_2.grad(&grads).unwrap();

        grad_1
            .to_data()
            .assert_approx_eq(&Data::from([[26.8063, -27.7870], [26.8063, -27.7870]]), 3);
        grad_2.to_data().assert_approx_eq(
            &Data::from([[9.222064, -39.123375], [-28.721354, 49.748356]]),
            3,
        );
    }
}
