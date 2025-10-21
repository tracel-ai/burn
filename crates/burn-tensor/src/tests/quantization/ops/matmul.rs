#[burn_tensor_testgen::testgen(q_matmul)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;
    use burn_tensor::{TensorPrimitive, Tolerance, ops::FloatElem, ops::QTensorOps};
    type FT = FloatElem<TestBackend>;

    #[test]
    #[ignore]
    fn test_matmul_vectors() {
        let tensor_1 = QTensor::<TestBackend, 2>::int8([[1.0, 2.0, 3.0, 6.35]]);
        let tensor_2 = QTensor::<TestBackend, 2>::int8([[12.7], [4.0], [5.0], [1.0]]);

        let tensor_3 = tensor_1.matmul(tensor_2);

        let expected = TensorData::from([[42.05]]);
        tensor_3
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::relative(2e-2));
    }

    #[test]
    #[ignore]
    fn test_matmul_2d() {
        let tensor_1 = QTensor::<TestBackend, 2>::int8([[1.0, 6.35], [2.0, 3.0], [1.0, 3.0]]);
        let tensor_2 = QTensor::<TestBackend, 2>::int8([[4.0, 8.0, 12.7], [2.0, 3.0, 6.0]]);
        let tensor_3 = tensor_1.matmul(tensor_2);

        let expected = TensorData::from([[16.7, 27.05, 50.8], [14., 25., 43.4], [10., 17., 30.7]]);
        tensor_3
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::relative(2e-2));
    }

    #[test]
    fn test_matmul_2d_aligned() {
        let tensor_1 = QTensor::<TestBackend, 2>::int8([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]);
        let tensor_2 = QTensor::<TestBackend, 2>::int8([
            [2.0, 0.0, 1.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ]);
        let tensor_3 = tensor_1.matmul(tensor_2);

        let expected = TensorData::from([
            [8.0, 7.0, 7.0, 4.0],
            [24.0, 19.0, 19.0, 8.0],
            [40.0, 31.0, 31.0, 12.0],
        ]);
        tensor_3
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::relative(2e-2));
    }

    #[test]
    fn test_matmul_2d_aligned_fused() {
        let tensor_1 = QTensor::<TestBackend, 2>::int8([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]);
        let tensor_2 = QTensor::<TestBackend, 2>::int8([
            [2.0, 0.0, 1.0, 0.0],
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ]);
        let tensor_3 = tensor_1.matmul(tensor_2);
        let tensor_4 = tensor_3 / 2.0;

        let expected = TensorData::from([
            [4.0, 3.5, 3.5, 2.0],
            [12.0, 9.5, 9.5, 4.0],
            [20.0, 15.5, 15.5, 6.0],
        ]);
        tensor_4
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::relative(2e-2));
    }

    #[test]
    #[ignore]
    fn test_matmul_3d() {
        let tensor_1 = QTensor::<TestBackend, 3>::int8([[[1.0, 6.35], [2.0, 3.0]]]);
        let tensor_2 = QTensor::<TestBackend, 3>::int8([[[12.7, 4.0], [2.0, 3.0]]]);

        let tensor_3 = tensor_1.matmul(tensor_2);
        let expected =
            TensorData::from([[[18.0, 28.0], [14.0, 23.0]], [[44.0, 26.0], [22.0, 19.0]]]);

        let expected = TensorData::from([[[25.4, 23.05], [31.4, 17.0]]]);
        tensor_3
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::relative(2e-2));
    }

    #[test]
    #[ignore]
    fn test_matmul_broadcast_4d() {
        let tensor_1 = QTensor::<TestBackend, 4>::int8([
            [[[1.0, 7.0], [2.0, 3.0]]],
            [[[2.0, 5.0], [6.0, 3.0]]],
        ]);
        let tensor_2 =
            QTensor::<TestBackend, 4>::int8([[[[9.0, 8.0], [1.0, 4.0]], [[2.0, 7.0], [3.0, 5.0]]]]);

        // [2, 1, 2, 2] @ [1, 2, 2, 2] -> [2, 2, 2, 2]
        let tensor_3 = tensor_1.matmul(tensor_2);
        let expected = TensorData::from([
            [[[16.0, 36.0], [21.0, 28.0]], [[23.0, 42.0], [13.0, 29.0]]],
            [[[23.0, 36.0], [57.0, 60.0]], [[19.0, 39.0], [21.0, 57.0]]],
        ]);

        tensor_3
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::relative(2e-2));
    }

    #[test]
    #[ignore]
    fn test_matmul_broadcast() {
        let tensor_1 = QTensor::<TestBackend, 3>::int8([[[1.0, 7.0], [2.0, 3.0]]]);
        let tensor_2 =
            QTensor::<TestBackend, 3>::int8([[[4.0, 7.0], [2.0, 3.0]], [[2.0, 5.0], [6.0, 3.0]]]);

        let tensor_3 = tensor_1.matmul(tensor_2);
        let expected =
            TensorData::from([[[18.0, 28.0], [14.0, 23.0]], [[44.0, 26.0], [22.0, 19.0]]]);

        tensor_3
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::relative(2e-2));
    }

    #[test]
    #[should_panic]
    fn should_panic_when_inner_dimensions_are_not_equal() {
        let tensor_1 = QTensor::<TestBackend, 2>::int8([[3., 3.], [4., 4.], [5., 5.], [6., 6.]]);
        let tensor_2 =
            QTensor::<TestBackend, 2>::int8([[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]]);

        let _ = tensor_1.matmul(tensor_2);
    }

    #[test]
    fn test_matmul_lhs_float_rhs_quantized() {
        // Simulates a typical workflow with linear layers (e.g., transformers), where the rhs
        // represents the weights. The lhs might be a float if a previous operation did not propagate
        // the quantization. We still want to perform an efficient matmul with quantized weights.
        let tensor_1 = TestTensor::<2>::from([
            [1.0, 6.35, 2.0, 3.0],
            [2.0, 3.0, 4.0, 5.0],
            [1.0, 3.0, 5.0, 7.0],
        ]);
        let tensor_2 = QTensor::<TestBackend, 2>::int8([
            [4.0, 8.0, 12.7, 1.6],
            [2.0, 3.0, 6.0, 4.0],
            [1.0, 5.0, 9.0, 2.5],
            [3.0, 7.0, 11.0, 0.5],
        ]);
        let tensor_3 = tensor_1.matmul(tensor_2);

        let expected = TensorData::from([
            [27.7, 58.05, 101.8, 33.5],
            [33., 80., 134.4, 27.7],
            [36., 91., 152.7, 29.6],
        ]);
        let output = tensor_3.into_data();
        output.assert_approx_eq::<FT>(&expected, Tolerance::default());

        // Default quantization scheme does not propagate quantization with matmul
        assert!(output.dtype.is_float());
    }
}
