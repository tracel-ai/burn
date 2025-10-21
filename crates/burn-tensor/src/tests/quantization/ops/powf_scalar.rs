#[burn_tensor_testgen::testgen(q_powf_scalar)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn should_support_powf_ops() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.powf_scalar(0.71);
        let expected = TensorData::from([[0.0, 1.0, 1.6358], [2.182, 2.6759, 3.1352]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_neg_power() {
        let tensor = QTensor::<TestBackend, 2>::int8([[1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.powf_scalar(-0.33);
        let expected =
            TensorData::from([[1.0, 1.0, 0.79553646], [0.695905, 0.6328783, 0.58794934]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_neg_values_with_even_power() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]]);

        let output = tensor.square();
        let expected = TensorData::from([[0., 1., 4.], [9., 16., 25.]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn should_support_neg_values_with_odd_power() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, -1.0, -2.0], [-3.0, -4.0, -4.0]]);

        let output = tensor.powf_scalar(3.0);
        let expected = TensorData::from([[0.0, -1.0, -8.0], [-27.0, -64.0, -64.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
