#[burn_tensor_testgen::testgen(tanh_activation)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Data, Tensor};

    #[test]
    fn test_tanh() {
        let data = Data::from([[1., 2.], [3., 4.]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = activation::tanh(tensor).to_data();

        let data_expected = Data::from([[0.7616, 0.9640], [0.9951, 0.9993]]);
        data_actual.assert_approx_eq(&data_expected, 4);
    }
}
