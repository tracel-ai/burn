#[burn_tensor_testgen::testgen(q_permute)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    // NOTE: we use affine quantization to reduce quantization errors for the given values range
    #[test]
    fn permute_float() {
        let tensor = QTensor::<TestBackend, 1>::int8_affine([
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        ])
        .reshape([2, 2, 4]);

        let permuted = tensor.clone().permute([2, 1, 0]);

        let expected = TensorData::from([
            [[0., 8.], [4., 12.]],
            [[1., 9.], [5., 13.]],
            [[2., 10.], [6., 14.]],
            [[3., 11.], [7., 15.]],
        ]);

        permuted
            .dequantize()
            .into_data()
            .assert_eq(&expected, false);

        // Test with negative axis
        let permuted = tensor.clone().permute([-1, 1, 0]);
        permuted
            .dequantize()
            .into_data()
            .assert_eq(&expected, false);

        // Test with the same axis
        let permuted = tensor.clone().permute([0, 1, 2]);
        permuted.into_data().assert_eq(&tensor.into_data(), true);
    }

    #[test]
    #[should_panic]
    fn edge_repeated_axes() {
        let tensor = QTensor::<TestBackend, 1>::int8_affine([
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        ])
        .reshape([2, 2, 4]);

        // Test with a repeated axis
        let _ = tensor.permute([0, 0, 1]);
    }

    #[test]
    #[should_panic]
    fn edge_out_of_bound_axis() {
        let tensor = QTensor::<TestBackend, 1>::int8_affine([
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
        ])
        .reshape([2, 2, 4]);

        // Test with an invalid axis
        let _ = tensor.permute([3, 0, 1]);
    }
}
