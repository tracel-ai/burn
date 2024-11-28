#[burn_tensor_testgen::testgen(q_sub)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    // NOTE: we use affine quantization to reduce quantization errors for range of input values
    #[test]
    fn should_support_sub_ops() {
        let tensor_1 = QTensor::<TestBackend, 2>::int8_affine([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor_2 = QTensor::<TestBackend, 2>::int8_affine([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]);

        let output = tensor_1 - tensor_2;
        let expected = TensorData::from([[-6.0, -6.0, -6.0], [-6.0, -6.0, -6.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn test_sub_broadcast() {
        let tensor_1 = QTensor::<TestBackend, 2>::int8_affine([[0.0, 1.0, 2.0]]);
        let tensor_2 = QTensor::<TestBackend, 2>::int8_affine([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]);

        let output = tensor_1 - tensor_2;
        let expected = TensorData::from([[-3.0, -3.0, -3.0], [-6.0, -6.0, -6.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_sub_scalar_ops() {
        let tensor = QTensor::<TestBackend, 2>::int8_affine([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let scalar = 2.0;

        let output = tensor - scalar;
        let expected = TensorData::from([[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }
}
