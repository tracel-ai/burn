#[burn_tensor_testgen::testgen(q_abs)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn should_support_abs_ops() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]]);

        let output = tensor.abs();

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), 1);
    }
}
