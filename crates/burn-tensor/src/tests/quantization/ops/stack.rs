#[burn_tensor_testgen::testgen(q_stack)]
mod tests {
    use super::*;
    use alloc::vec;
    use burn_tensor::{Tensor, TensorData};

    // NOTE: we use affine quantization to reduce quantization errors for range of input values
    #[test]
    fn should_support_stack_ops_2d_dim0() {
        let tensor_1 = QTensor::<TestBackend, 2>::int8_affine([[1.0, 2.0, 3.0]]);
        let tensor_2 = QTensor::<TestBackend, 2>::int8_affine([[4.0, 5.0, 6.0]]);

        let output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 0);
        let expected = TensorData::from([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_stack_ops_2d_dim1() {
        let tensor_1 = QTensor::<TestBackend, 2>::int8_affine([[1.0, 2.0, 3.0]]);
        let tensor_2 = QTensor::<TestBackend, 2>::int8_affine([[4.0, 5.0, 6.0]]);

        let output = Tensor::stack::<3>(vec![tensor_1, tensor_2], 1);
        let expected = TensorData::from([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_support_stack_ops_3d() {
        let tensor_1 =
            QTensor::<TestBackend, 3>::int8_affine([[[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]]]);
        let tensor_2 =
            QTensor::<TestBackend, 3>::int8_affine([[[4.0, 5.0, 6.0]], [[6.0, 5.0, 4.0]]]);

        let output = Tensor::stack::<4>(vec![tensor_1, tensor_2], 0);
        let expected = TensorData::from([
            [[[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]]],
            [[[4.0, 5.0, 6.0]], [[6.0, 5.0, 4.0]]],
        ]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_dimensions_are_not_the_same() {
        let tensor_1 = QTensor::<TestBackend, 2>::int8_affine([[1.0, 2.0, 3.0]]);
        let tensor_2 = QTensor::<TestBackend, 2>::int8_affine([[4.0, 5.0]]);

        let output: TestTensor<3> = Tensor::stack(vec![tensor_1, tensor_2], 0);
    }

    #[test]
    #[should_panic]
    fn should_panic_when_stack_exceeds_dimension() {
        let tensor_1 =
            QTensor::<TestBackend, 3>::int8_affine([[[1.0, 2.0, 3.0]], [[3.0, 2.0, 1.0]]]);
        let tensor_2 = QTensor::<TestBackend, 3>::int8_affine([[[4.0, 5.0, 6.0]]]);

        let output: TestTensor<4> = TestTensor::stack(vec![tensor_1, tensor_2], 3);
    }
}
