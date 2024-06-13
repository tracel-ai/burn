#[burn_tensor_testgen::testgen(tanh_activation)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Tensor, TensorData};

    #[test]
    fn test_tanh() {
        let tensor = TestTensor::from([[1., 2.], [3., 4.]]);

        let data_actual = activation::tanh(tensor).into_data();

        let data_expected = TensorData::from([[0.7616, 0.9640], [0.9951, 0.9993]]);
        data_actual.assert_approx_eq(&data_expected, 4);
    }
}
