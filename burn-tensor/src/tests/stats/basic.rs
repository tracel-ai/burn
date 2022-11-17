#[burn_tensor_testgen::testgen(stats)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_var() {
        let data = Data::from([[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.var(1).into_data();

        let data_expected = Data::from([[2.4892], [15.3333]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }
}
