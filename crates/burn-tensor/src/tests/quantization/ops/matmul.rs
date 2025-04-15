#[burn_tensor_testgen::testgen(q_matmul)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn test_matmul_vectors() {
        let tensor_1 = QTensor::<TestBackend, 2>::int8([[1.0, 2.0, 3.0, 6.35]]);
        let tensor_2 = QTensor::<TestBackend, 2>::int8([[12.7], [4.0], [5.0], [1.0]]);

        let tensor_3 = tensor_1.matmul(tensor_2);
        let expected = TensorData::from([[half::f16::from_f32(42.05)]]);
        tensor_3.into_data().assert_approx_eq_diff(&expected, 0.3);
    }

    #[test]
    fn test_matmul_2d() {
        let tensor_1 = QTensor::<TestBackend, 2>::int8([[1.0, 6.35], [2.0, 3.0], [1.0, 3.0]]);
        let tensor_2 = QTensor::<TestBackend, 2>::int8([[4.0, 8.0, 12.7], [2.0, 3.0, 6.0]]);
        let tensor_3 = tensor_1.matmul(tensor_2);

        let expected = TensorData::from([[16.7, 27.05, 50.8], [14., 25., 43.4], [10., 17., 30.7]]);
        tensor_3.into_data().assert_approx_eq_diff(&expected, 0.3);
    }

    #[test]
    fn test_matmul_3d() {
        let tensor_1 = QTensor::<TestBackend, 3>::int8([[[1.0, 6.35], [2.0, 3.0]]]);
        let tensor_2 = QTensor::<TestBackend, 3>::int8([[[12.7, 4.0], [2.0, 3.0]]]);

        let tensor_3 = tensor_1.matmul(tensor_2);
        let expected = TensorData::from([[[25.4, 23.05], [31.4, 17.0]]]);
        tensor_3.into_data().assert_approx_eq_diff(&expected, 0.3);
    }

    #[test]
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

        tensor_3.into_data().assert_approx_eq_diff(&expected, 0.3);
    }

    #[test]
    fn test_matmul_transposed() {
        let tensor_1 = QTensor::<TestBackend, 2>::int8([
            [0., 1., 2., 3.],
            [4., 5., 6., 7.],
            [8., 9., 10., 11.],
            [12., 13., 14., 15.],
        ]);

        let tensor_3 = tensor_1.clone().matmul(tensor_1.transpose());

        tensor_3.into_data().assert_approx_eq(
            &TensorData::from([
                [14., 38., 62., 86.],
                [38., 126., 214., 302.],
                [62., 214., 366., 518.],
                [86., 302., 518., 734.],
            ]),
            1,
        );
    }

    #[test]
    fn test_matmul_broadcast() {
        let tensor_1 = QTensor::<TestBackend, 3>::int8([[[1.0, 7.0], [2.0, 3.0]]]);
        let tensor_2 =
            QTensor::<TestBackend, 3>::int8([[[4.0, 7.0], [2.0, 3.0]], [[2.0, 5.0], [6.0, 3.0]]]);

        let tensor_3 = tensor_1.matmul(tensor_2);
        let expected =
            TensorData::from([[[18.0, 28.0], [14.0, 23.0]], [[44.0, 26.0], [22.0, 19.0]]]);

        tensor_3.into_data().assert_approx_eq_diff(&expected, 0.3);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_inner_dimensions_are_not_equal() {
        let tensor_1 = QTensor::<TestBackend, 2>::int8([[3., 3.], [4., 4.], [5., 5.], [6., 6.]]);
        let tensor_2 =
            QTensor::<TestBackend, 2>::int8([[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]]);

        let _ = tensor_1.matmul(tensor_2);
    }
}
