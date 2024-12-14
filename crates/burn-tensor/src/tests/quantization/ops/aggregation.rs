#[burn_tensor_testgen::testgen(q_aggregation)]
mod tests {
    use super::*;
    use burn_tensor::TensorData;

    #[test]
    fn test_should_mean() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.mean();

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([15.0 / 6.0]), 1);
    }

    #[test]
    fn test_should_sum() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.sum();

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([15.0]), 1);
    }

    #[test]
    fn test_should_mean_last_dim() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.mean_dim(1);
        let expected = TensorData::from([[3.0 / 3.0], [12.0 / 3.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn test_should_sum_last_dim() {
        let tensor = QTensor::<TestBackend, 2>::int8([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.sum_dim(1);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[3.0], [12.0]]), 1);
    }

    #[test]
    fn test_should_sum_first_dim() {
        let tensor = QTensor::<TestBackend, 2>::int8([[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]]);

        let output = tensor.sum_dim(0);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[7.0, 3.0, 5.0]]), 1);
    }

    #[test]
    fn test_should_mean_first_dim() {
        let tensor = QTensor::<TestBackend, 2>::int8([[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]]);

        let output = tensor.mean_dim(0);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[7.0 / 2.0, 3.0 / 2.0, 5.0 / 2.0]]), 1);
    }

    #[test]
    fn test_should_sum_mid_dim_3d_non_contiguous_1() {
        let tensor = QTensor::<TestBackend, 3>::int8([
            [[2.0, 4.0, 1.0], [7.0, -5.0, 3.0]],
            [[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]],
        ]);

        let output = tensor.swap_dims(0, 2).sum_dim(1);

        // Precision 1 to approximate de/quantization errors
        output.dequantize().into_data().assert_approx_eq(
            &TensorData::new(vec![9.0, 7.0, -1.0, 3.0, 4.0, 5.0], [3, 1, 2]),
            1,
        );
    }

    #[test]
    fn test_should_sum_mid_dim_3d_non_contiguous_2() {
        let tensor = QTensor::<TestBackend, 3>::int8([
            [[2.0, 4.0, 1.0], [7.0, -5.0, 3.0]],
            [[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]],
        ]);

        let output = tensor.swap_dims(0, 1).sum_dim(1);

        // Precision 1 to approximate de/quantization errors
        output.dequantize().into_data().assert_approx_eq(
            &TensorData::new(vec![5.0, 5.0, 3.0, 11.0, -3.0, 6.0], [2, 1, 3]),
            1,
        );
    }

    #[test]
    fn test_prod_float() {
        // NOTE: we use affine quantization to reduce quantization errors since `prod()` amplifies the error
        let tensor = QTensor::<TestBackend, 2>::int8_affine([[2.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.prod();

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([240.0]), 3);

        let tensor_with_zero =
            QTensor::<TestBackend, 2>::int8_affine([[2.0, 0.0, 2.0], [3.0, 4.0, 5.0]]);
        let output = tensor_with_zero.prod();

        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([0.0]), 5);
    }

    #[test]
    fn test_prod_dim_float() {
        // NOTE: we use affine quantization to reduce quantization errors since `prod()` amplifies the error
        let tensor = QTensor::<TestBackend, 2>::int8_affine([[2.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.prod_dim(1);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[4.0], [60.0]]), 1);

        let tensor_with_zero =
            QTensor::<TestBackend, 2>::int8_affine([[2.0, 0.0, 2.0], [3.0, 4.0, 5.0]]);
        let output = tensor_with_zero.prod_dim(1);
        let expected = TensorData::from([[0.0], [60.0]]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }
}
