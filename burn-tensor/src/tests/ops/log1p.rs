#[burn_tensor_testgen::testgen(log1p)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_exp_log1p() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.log1p().into_data();

        let data_expected = Data::from([
            [0.0, core::f32::consts::LN_2, 1.0986],
            [1.3862, 1.6094, 1.7917],
        ]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }
}
