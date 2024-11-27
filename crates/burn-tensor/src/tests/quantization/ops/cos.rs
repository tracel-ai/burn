#[burn_tensor_testgen::testgen(q_cos)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn should_support_cos_ops() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.cos();
        let expected = TensorData::from([[1.0, 0.5403, -0.4161], [-0.9899, -0.6536, 0.2836]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }
}
