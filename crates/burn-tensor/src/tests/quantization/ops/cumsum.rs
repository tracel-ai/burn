#[burn_tensor_testgen::testgen(q_cumsum)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn test_should_support_cumsum_ops() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.cumsum(0);
        let expected = TensorData::from([[0.0, 1.0, 2.0], [3.0, 5.0, 7.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }
}
