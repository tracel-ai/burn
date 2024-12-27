#[burn_tensor_testgen::testgen(q_select)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    // NOTE: we use affine quantization to reduce quantization errors for range of input values
    #[test]
    fn should_select_1d() {
        let tensor = QTensor::<TestBackend, 1>::int8_affine([0.0, 1.0, 2.0, 3.0]);
        let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &Default::default());

        let output = tensor.select(0, indices);
        let expected = TensorData::from([1.0, 1.0, 0.0, 1.0, 2.0]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_2d_dim0_same_num_dim() {
        let tensor = QTensor::<TestBackend, 2>::int8_affine([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let indices = TestTensorInt::from_data(([1, 0]), &Default::default());

        let output = tensor.select(0, indices);
        let expected = TensorData::from([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_2d_dim0_more_num_dim() {
        let tensor = QTensor::<TestBackend, 2>::int8_affine([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let indices = TestTensorInt::from_data([1, 0, 1, 1], &Default::default());

        let output = tensor.select(0, indices);
        let expected = TensorData::from([
            [3.0, 4.0, 5.0],
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0],
        ]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_2d_dim1() {
        let tensor = QTensor::<TestBackend, 2>::int8_affine([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &Default::default());

        let output = tensor.select(1, indices);
        let expected = TensorData::from([[1.0, 1.0, 0.0, 1.0, 2.0], [4.0, 4.0, 3.0, 4.0, 5.0]]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_select_assign_1d() {
        let tensor = QTensor::<TestBackend, 1>::int8_affine([0.0, 1.0, 2.0]);
        let values = QTensor::<TestBackend, 1>::int8_affine([5.0, 4.0, 3.0, 2.0, 1.0]);
        let indices =
            TestTensorInt::from_data(TensorData::from([1, 1, 0, 1, 2]), &Default::default());

        let output = tensor.select_assign(0, indices, values);
        let expected = TensorData::from([3.0, 12.0, 3.0]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_select_assign_2d_dim0() {
        let tensor = QTensor::<TestBackend, 2>::int8_affine([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let values = tensor.clone();
        let indices = TestTensorInt::from_data(TensorData::from([1, 0]), &Default::default());

        let output = tensor.select_assign(0, indices, values);
        let expected = TensorData::from([[3.0, 5.0, 7.0], [3.0, 5.0, 7.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_select_assign_2d_dim1() {
        let tensor = QTensor::<TestBackend, 2>::int8_affine([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let values = tensor.clone();
        let indices = TestTensorInt::from_data(TensorData::from([1, 0, 2]), &Default::default());

        let output = tensor.select_assign(1, indices, values);
        let expected = TensorData::from([[1.0, 1.0, 4.0], [7.0, 7.0, 10.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    #[should_panic]
    fn should_select_panic_invalid_dimension() {
        let tensor = QTensor::<TestBackend, 2>::int8_affine([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let indices = TestTensorInt::from_data([1, 1, 0, 1, 2], &Default::default());

        tensor.select(10, indices);
    }
}
