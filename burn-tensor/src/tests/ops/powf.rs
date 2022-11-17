#[burn_tensor_testgen::testgen(powf)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_powf_ops() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.powf(0.71).into_data();

        let data_expected = Data::from([[0.0, 1.0, 1.6358], [2.182, 2.6759, 3.1352]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }
}
