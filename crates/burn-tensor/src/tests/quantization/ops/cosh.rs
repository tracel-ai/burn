#[burn_tensor_testgen::testgen(q_cosh)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn should_support_cosh_ops() {
        // NOTE: we use affine quantization to reduce quantization errors for range of input values
        let tensor = QTensor::<TestBackend, 2>::int8_affine([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.cosh();
        let expected = TensorData::from([[1.0000, 1.5431, 3.7622], [10.0677, 27.3082, 74.2100]]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 3);
    }
}
