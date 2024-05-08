#[burn_tensor_testgen::testgen(cumsum_dim)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn should_cumsum_over_dim() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());

        let dim = 1;

        let data_actual = tensor.cumsum_dim(dim).into_data();
        let data_expected = Data::from([[0.0, 1.0, 3.0], [3.0, 7.0, 12.0], [6.0, 13.0, 21.0]]);

        data_expected.assert_approx_eq(&data_actual, 3);
    }
}
