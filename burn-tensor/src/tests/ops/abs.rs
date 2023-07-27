#[burn_tensor_testgen::testgen(abs)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_abs_ops() {
        let data = Data::from([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.abs().into_data();

        let data_expected = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }
}
