#[burn_tensor_testgen::testgen(powf)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn should_support_powf_ops() {
        let data = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());
        let pow = Data::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor_pow = Tensor::<TestBackend, 2>::from_data(pow, &Default::default());
        let data_actual = tensor.powf(tensor_pow).into_data();
        let data_expected = Data::from([[1.0, 1.0, 4.0], [27.0, 256.0, 3125.0]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_neg_power() {
        let data = Data::from([[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());
        let pow = Data::from([[-0.95, -0.67, -0.45], [-0.24, -0.5, -0.6]]);
        let tensor_pow = Tensor::<TestBackend, 2>::from_data(pow, &Default::default());

        let data_actual = tensor.powf(tensor_pow).into_data();

        let data_expected = Data::from([[1., 1., 0.73204285], [0.76822936, 0.5, 0.38073079]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_neg_values_with_even_power() {
        let data = Data::from([[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());
        let pow = Data::from([[2.0, 2.0, 4.0], [4.0, 4.0, 2.0]]);
        let tensor_pow = Tensor::<TestBackend, 2>::from_data(pow, &Default::default());
        let data_actual = tensor.powf(tensor_pow).into_data();
        let data_expected = Data::from([[0.0, 1.0, 16.0], [81.0, 256.0, 25.0]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn should_support_neg_values_with_odd_power() {
        let data = Data::from([[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data, &Default::default());
        let pow = Data::from([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]);
        let tensor_pow = Tensor::<TestBackend, 2>::from_data(pow, &Default::default());
        let data_actual = tensor.powf(tensor_pow).into_data();

        let data_expected = Data::from([[0.0, -1.0, -8.0], [-27.0, -64.0, -125.0]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }
}
