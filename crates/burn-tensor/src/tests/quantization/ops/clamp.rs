#[burn_tensor_testgen::testgen(q_clamp)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn clamp_min() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.clamp_min(2.0);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[2.0, 2.0, 2.0], [3.0, 4.0, 5.0]]), 1);
    }

    #[test]
    fn clamp_max() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.clamp_max(2.0);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[0.0, 1.0, 2.0], [2.0, 2.0, 2.0]]), 1);
    }

    #[test]
    fn clamp_min_max() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.clamp(1.0, 4.0);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[1.0, 1.0, 2.0], [3.0, 4.0, 4.0]]), 1);
    }
}
