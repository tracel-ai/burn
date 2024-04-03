#[burn_tensor_testgen::testgen(ad_log_sigmoid)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Data};

    #[test]
    fn should_diff_log_sigmoid() {
        let data = Data::<f32, 2>::from([[0.8762, -0.1423], [-300., 200.]]);

        let device = Default::default();
        let tensor_1 = TestAutodiffTensor::from_data(data, &device).require_grad();
        let tensor_2 = activation::log_sigmoid(tensor_1.clone());
        let grads = tensor_2.backward();

        let grad = tensor_1.grad(&grads).unwrap();

        grad.to_data()
            .assert_approx_eq(&Data::from([[0.293966, 0.535515], [1.000000, 0.000000]]), 4);
    }
}
