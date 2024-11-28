#[burn_tensor_testgen::testgen(q_sqrt)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;
    use core::f32::consts::SQRT_2;

    #[test]
    fn should_support_sqrt_ops() {
        // NOTE: we use affine quantization to reduce quantization errors for range of input values
        let tensor = QTensor::<TestBackend, 2>::int8_affine([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.sqrt();
        let expected = TensorData::from([[0.0, 1.0, SQRT_2], [1.73205, 2.0, 2.2360]]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 3);
    }
}
