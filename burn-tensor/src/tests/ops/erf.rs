#[burn_tensor_testgen::testgen(erf)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_erf_ops() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.erf().into_data();

        let data_expected = Data::from([[0.0000, 0.8427, 0.9953], [1.0000, 1.0000, 1.0000]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_erf_ops_with_negative_number() {
        let data = Data::from([[-0.056, -0.043, -0.089], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.erf().into_data();

        let data_expected = Data::from([
            [-0.06312324, -0.048490416, -0.10016122],
            [1.0000, 1.0000, 1.0000],
        ]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }
}
