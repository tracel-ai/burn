#[burn_tensor_testgen::testgen(q_tan)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn should_support_tan_ops() {
        // NOTE: we use affine quantization to reduce quantization errors for range of input values
        let tensor = QTensor::<TestBackend, 2>::int8_affine([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.tan();
        let expected = TensorData::from([[0.0, 1.5574, -2.1850], [-0.1425, 1.1578, -3.3805]]);

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 3);
    }
}
