#[burn_tensor_testgen::testgen(q_repeat_dim)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    // NOTE: we use affine quantization to reduce quantization errors for range of input values
    #[test]
    fn should_support_repeat_ops() {
        let tensor = QTensor::<TestBackend, 2>::int8_affine([[0.0, 1.0, 2.0, 3.0]]);

        let output = tensor.repeat_dim(0, 4);
        let expected = TensorData::from([
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
        ]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_repeat_on_dims_larger_than_1() {
        let tensor = QTensor::<TestBackend, 1>::int8_affine([
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        ])
        .reshape([4, 2, 2]);

        let output = tensor.repeat_dim(2, 2);
        let expected = TensorData::from([
            [[0., 1., 0., 1.], [2., 3., 2., 3.]],
            [[4., 5., 4., 5.], [6., 7., 6., 7.]],
            [[8., 9., 8., 9.], [10., 11., 10., 11.]],
            [[12., 13., 12., 13.], [14., 15., 14., 15.]],
        ]);

        output.dequantize().into_data().assert_eq(&expected, false);
    }
}
