#[burn_tensor_testgen::testgen(softplus)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Data, Tensor};

    #[test]
    fn test_softplus_d2() {
        let data = Data::from([[-0.4240, -0.9574, -0.2215], [-0.5767, 0.7218, -0.1620]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual_beta1 = activation::softplus(tensor.clone(), 1.0).to_data();
        let data_expected_beta1 = Data::from([[0.5034, 0.3249, 0.5885], [0.4458, 1.1178, 0.6154]]);
        data_actual_beta1.assert_approx_eq(&data_expected_beta1, 4);

        let data_actual_beta2 = activation::softplus(tensor, 2.0).to_data();
        let data_expected_beta2 = Data::from([[0.1782, 0.0687, 0.2480], [0.1371, 0.8277, 0.2721]]);
        data_actual_beta2.assert_approx_eq(&data_expected_beta2, 4);
    }
}
