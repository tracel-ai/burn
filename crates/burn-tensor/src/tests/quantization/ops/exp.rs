#[burn_tensor_testgen::testgen(q_exp)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn should_support_exp_ops() {
        // NOTE: we use affine quantization to reduce quantization errors since `exp()` amplifies the error
        let tensor = QTensor::<TestBackend, 2>::int8_affine([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.exp();
        let expected = TensorData::from([[1.0, 2.71830, 7.3891], [20.0855, 54.5981, 148.4132]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }
}
