#[burn_tensor_testgen::testgen(ad_recip)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[test]
    fn should_diff_recip() {
        let data = Data::from([2.0, 5.0, 0.4]);

        let tensor = TestAutodiffTensor::from_data_devauto(data).require_grad();
        let tensor_out = tensor.clone().recip();

        let grads = tensor_out.backward();
        let grad = tensor.grad(&grads).unwrap();

        assert_eq!(tensor_out.into_data(), Data::from([0.5, 0.2, 2.5]));
        grad.to_data()
            .assert_approx_eq(&Data::from([-0.25, -0.04, -6.25]), 3);
    }
}
