#[burn_tensor_testgen::testgen(sigmoid)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Data, Tensor};

    #[test]
    fn test_sigmoid() {
        let tensor = TestTensor::from([[1.0, 7.0], [13.0, -3.0]]);

        let data_actual = activation::sigmoid(tensor).into_data();

        let data_expected = Data::from([[0.7311, 0.9991], [1.0, 0.0474]]);
        data_actual.assert_approx_eq(&data_expected, 4);
    }

    #[test]
    fn test_sigmoid_overflow() {
        let tensor = TestTensor::from([f32::MAX, f32::MIN]);

        let data_actual = activation::sigmoid(tensor).into_data();

        let data_expected = Data::from([1.0, 0.0]);
        data_actual.assert_approx_eq(&data_expected, 4);
    }
}
