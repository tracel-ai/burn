#[burn_tensor_testgen::testgen(q_sin)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn should_support_sin_ops() {
        // NOTE: we use affine quantization to reduce quantization errors for range of input values
        let tensor = QTensor::<TestBackend, 2>::int8_affine([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.sin();
        let expected = TensorData::from([[0.0, 0.8414, 0.9092], [0.1411, -0.7568, -0.9589]]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 3);
    }
}
