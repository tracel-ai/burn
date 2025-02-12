#[burn_tensor_testgen::testgen(q_topk)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    // NOTE: we use affine quantization to reduce quantization errors for range of input values
    #[test]
    fn test_topk_1d() {
        let tensor = QTensor::<TestBackend, 1>::int8_affine([1.0, 2.0, 3.0, 4.0, 5.0]);

        let values = tensor.topk(3, /*dim*/ 0);
        let expected = TensorData::from([5., 4., 3.]);

        values
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 3);
    }

    #[test]
    fn test_topk() {
        let tensor = QTensor::<TestBackend, 3>::int8_affine([
            [[1., 4., 7.], [2., 5., 6.]],
            [[3., 0., 9.], [8., 2., 7.]],
        ]);

        let values = tensor.topk(2, /*dim*/ 2);
        let expected = TensorData::from([[[7., 4.], [6., 5.]], [[9., 3.], [8., 7.]]]);

        // Precision 1 to approximate de/quantization errors
        values
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn test_topk_with_indices() {
        // 1D
        let tensor = QTensor::<TestBackend, 1>::int8_affine([1.0, 2.0, 3.0, 4.0, 5.0]);

        let (values, indices) = tensor.topk_with_indices(3, /*dim*/ 0);

        let values_expected = TensorData::from([5., 4., 3.]);
        values
            .dequantize()
            .into_data()
            .assert_eq(&values_expected, false);

        let indices_expected = TensorData::from([4, 3, 2]);
        indices.into_data().assert_eq(&indices_expected, false);
    }

    #[test]
    fn test_topk_with_indices_3d() {
        // 3D
        let tensor = QTensor::<TestBackend, 3>::int8_affine([
            [[1., 4., 7.], [2., 5., 6.]],
            [[3., 0., 9.], [8., 2., 7.]],
        ]);

        let (values, indices) = tensor.topk_with_indices(2, /*dim*/ 2);

        let values_expected = TensorData::from([[[7., 4.], [6., 5.]], [[9., 3.], [8., 7.]]]);

        // Precision 1 to approximate de/quantization errors
        values
            .dequantize()
            .into_data()
            .assert_approx_eq(&values_expected, 1);

        let indices_expected = TensorData::from([[[2, 1], [2, 1]], [[2, 0], [0, 2]]]);

        indices.into_data().assert_eq(&indices_expected, false);
    }
}
