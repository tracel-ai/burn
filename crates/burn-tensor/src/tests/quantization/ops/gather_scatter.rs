#[burn_tensor_testgen::testgen(q_gather_scatter)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn should_gather_1d_dim0() {
        let tensor = QTensor::<TestBackend, 1>::int8([0.0, 1.0, 2.0]);
        let indices = TestTensorInt::from_ints([1, 1, 0, 1, 2], &Default::default());

        let output = tensor.gather(0, indices);

        // Precision 1 to approximate de/quantization errors
        output.dequantize().into_data().assert_approx_eq::<FT>(
            &TensorData::from([1.0, 1.0, 0.0, 1.0, 2.0]),
            Tolerance::absolute(1e-1),
        );
    }

    #[test]
    fn should_gather_2d_dim0() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let indices = TestTensorInt::from_ints([[0, 1, 0], [1, 0, 1]], &Default::default());

        let output = tensor.gather(0, indices);

        // Precision 1 to approximate de/quantization errors
        output.dequantize().into_data().assert_approx_eq::<FT>(
            &TensorData::from([[0.0, 4.0, 2.0], [3.0, 1.0, 5.0]]),
            Tolerance::absolute(1e-1),
        );
    }

    #[test]
    fn should_gather_2d_dim1() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let indices = TestTensorInt::from_ints([[2, 1, 0, 0], [2, 0, 1, 2]], &Default::default());

        let output = tensor.gather(1, indices);

        // Precision 1 to approximate de/quantization errors
        output.dequantize().into_data().assert_approx_eq::<FT>(
            &TensorData::from([[2.0, 1.0, 0.0, 0.0], [5.0, 3.0, 4.0, 5.0]]),
            Tolerance::absolute(1e-1),
        );
    }

    #[test]
    fn should_gather_3d_dim1() {
        let tensor = QTensor::<TestBackend, 3>::int8([
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ]);
        let indices = TestTensorInt::from_ints(
            [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 1, 1]]],
            &Default::default(),
        );

        let output = tensor.gather(1, indices);
        let expected = TensorData::from([
            [[3.0, 1.0, 2.0], [0.0, 4.0, 2.0]],
            [[6.0, 7.0, 11.0], [6.0, 10.0, 11.0]],
        ]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::absolute(1e-1));
    }

    #[test]
    fn should_gather_2d_only_1dim() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let indices = TestTensorInt::<2>::from_ints([[1, 2]], &Default::default()).reshape([2, 1]);

        let output = tensor.gather(1, indices);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq::<FT>(&TensorData::from([[1.0], [5.0]]), Tolerance::absolute(1e-1));
    }

    #[test]
    fn should_scatter_1d() {
        let tensor = QTensor::<TestBackend, 1>::int8([0.0, 0.0, 0.0]);
        let values = QTensor::<TestBackend, 1>::int8([5.0, 4.0, 3.0]);
        let indices = TestTensorInt::from_ints([1, 0, 2], &Default::default());

        let output = tensor.scatter_add(0, indices, values);

        // Precision 1 to approximate de/quantization errors
        output.dequantize().into_data().assert_approx_eq::<FT>(
            &TensorData::from([4.0, 5.0, 3.0]),
            Tolerance::absolute(1e-1),
        );
    }

    #[test]
    fn should_scatter_2d_dim0() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let values = QTensor::<TestBackend, 2>::int8([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let indices = TestTensorInt::from_ints([[1, 0, 1], [1, 1, 0]], &Default::default());

        let output = tensor.scatter_add(0, indices, values);

        // Precision 1 to approximate de/quantization errors
        output.dequantize().into_data().assert_approx_eq::<FT>(
            &TensorData::from([[0.0, 2.0, 6.0], [5.0, 5.0, 3.0]]),
            Tolerance::absolute(1e-1),
        );
    }

    #[test]
    fn should_scatter_2d_dim1() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let values = QTensor::<TestBackend, 2>::int8([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let indices = TestTensorInt::from_ints([[1, 0, 2], [1, 2, 0]], &Default::default());

        let output = tensor.scatter_add(1, indices, values);

        // Precision 1 to approximate de/quantization errors
        output.dequantize().into_data().assert_approx_eq::<FT>(
            &TensorData::from([[2.0, 1.0, 3.0], [6.0, 4.0, 5.0]]),
            Tolerance::absolute(1e-1),
        );
    }

    #[test]
    fn should_scatter_3d_dim1() {
        let tensor = QTensor::<TestBackend, 3>::int8([
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        ]);
        let values = QTensor::<TestBackend, 3>::int8([
            [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
            [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
        ]);
        let indices = TestTensorInt::from_ints(
            [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 1, 1]]],
            &Default::default(),
        );

        let output = tensor.scatter_add(1, indices, values);
        let expected = TensorData::from([
            [[15.0, 14.0, 33.0], [15.0, 20.0, 5.0]],
            [[45.0, 26.0, 8.0], [9.0, 32.0, 54.0]],
        ]);

        // Set higher tolerance (0.2) due to larger de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::absolute(2e-1));
    }

    #[test]
    fn should_scatter_2d_dim1_diff_shape() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let values = QTensor::<TestBackend, 2>::int8([[1.0], [4.0]]);
        let indices = TestTensorInt::from_ints([[1], [2]], &Default::default());

        let output = tensor.scatter_add(1, indices, values);

        // Precision 1 to approximate de/quantization errors
        output.dequantize().into_data().assert_approx_eq::<FT>(
            &TensorData::from([[0.0, 1.0, 0.0], [0.0, 0.0, 4.0]]),
            Tolerance::absolute(1e-1),
        );
    }

    #[test]
    #[should_panic]
    fn scatter_should_panic_on_mismatch_of_shapes() {
        let tensor = QTensor::<TestBackend, 1>::int8([0.0, 0.0, 0.0]);
        let values = QTensor::<TestBackend, 1>::int8([1.0, 4.0]);
        let indices = TestTensorInt::from_ints([1, 0, 2], &Default::default());

        tensor.scatter_add(0, indices, values);
    }
}
