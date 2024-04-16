#[burn_tensor_testgen::testgen(log_sigmoid)]
mod tests {
    use super::*;
    use burn_tensor::{activation, Data, Tensor};

    #[test]
    fn test_log_sigmoid() {
        let tensor = TestTensor::from([[1.0, 7.0], [13.0, -3.0]]);

        let data_actual = activation::log_sigmoid(tensor).into_data();

        let data_expected = Data::from([[-3.132617e-1, -9.114665e-4], [-2.260327e-6, -3.0485873]]);
        data_actual.assert_approx_eq(&data_expected, 4);
    }

    #[test]
    fn test_log_sigmoid_numerical_stability() {
        let tensor = TestTensor::from([300.0, -300.0]);

        let data_actual = activation::log_sigmoid(tensor).into_data();

        // For large negative values, the previous implementation −log(1 + exp(−x)) would give -inf
        let data_expected = Data::from([0.0, -300.0]);
        data_actual.assert_approx_eq(&data_expected, 4);

        let tensor = TestTensor::from([f32::MAX, f32::MIN]);

        let data_actual = activation::log_sigmoid(tensor).into_data();

        let data_expected = Data::from([0.0, f32::MIN]);
        data_actual.assert_approx_eq(&data_expected, 4);
    }
}
