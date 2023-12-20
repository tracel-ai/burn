#[burn_tensor_testgen::testgen(recip)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_recip_ops() {
        let data = Data::from([[0.5, 1.0, 2.0], [3.0, -4.0, -5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data_devauto(data);

        let data_actual = tensor.recip().into_data();

        let data_expected = Data::from([[2.0, 1.0, 0.5], [0.33333, -0.25, -0.2]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }
}
