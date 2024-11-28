#[burn_tensor_testgen::testgen(q_recip)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn should_support_recip_ops() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.5, 1.0, 2.0], [3.0, -4.0, -5.0]]);

        let output = tensor.recip();
        let expected = TensorData::from([[2.0, 1.0, 0.5], [0.33333, -0.25, -0.2]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }
}
