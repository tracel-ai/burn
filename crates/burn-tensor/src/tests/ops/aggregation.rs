#[burn_tensor_testgen::testgen(aggregation)]
mod tests {
    use super::*;
    use burn_tensor::{Shape, Tensor, TensorData};

    #[test]
    fn test_should_mean() {
        let tensor = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.mean();
        let expected = TensorData::from([15.0 / 6.0]);

        output.into_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn test_should_mean_int() {
        let tensor = TestTensorInt::<2>::from([[2, 2, 2], [3, 4, 5]]);

        let output = tensor.mean();

        output.into_data().assert_eq(&TensorData::from([3]), false);
    }

    #[test]
    fn test_should_sum() {
        let tensor = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.sum();

        output
            .into_data()
            .assert_eq(&TensorData::from([15.0]), false);
    }

    #[test]
    fn test_should_sum_int() {
        let tensor = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

        let output = tensor.sum();

        output.into_data().assert_eq(&TensorData::from([15]), false);
    }

    #[test]
    fn test_should_mean_last_dim() {
        let tensor = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.mean_dim(1);
        let expected = TensorData::from([[3.0 / 3.0], [12.0 / 3.0]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }

    #[test]
    fn test_should_sum_last_dim() {
        let tensor = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.sum_dim(1);

        output
            .into_data()
            .assert_eq(&TensorData::from([[3.0], [12.0]]), false);
    }

    #[test]
    fn test_should_mean_last_dim_int() {
        let tensor = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

        let output = tensor.mean_dim(1);

        output
            .into_data()
            .assert_eq(&TensorData::from([[1], [4]]), false);
    }

    #[test]
    fn test_should_sum_last_dim_int() {
        let tensor = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

        let output = tensor.sum_dim(1);

        output
            .into_data()
            .assert_eq(&TensorData::from([[3], [12]]), false);
    }

    #[test]
    fn test_should_sum_first_dim() {
        let tensor = TestTensor::<2>::from([[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]]);

        let output = tensor.sum_dim(0);

        output
            .into_data()
            .assert_eq(&TensorData::from([[7.0, 3.0, 5.0]]), false);
    }

    #[test]
    fn test_should_mean_first_dim() {
        let tensor = TestTensor::<2>::from([[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]]);

        let output = tensor.mean_dim(0);

        output.into_data().assert_eq(
            &TensorData::from([[7.0 / 2.0, 3.0 / 2.0, 5.0 / 2.0]]),
            false,
        );
    }

    #[test]
    fn test_should_sum_mid_dim_3d_non_contiguous_1() {
        let tensor = TestTensor::<3>::from([
            [[2.0, 4.0, 1.0], [7.0, -5.0, 3.0]],
            [[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]],
        ]);

        let output = tensor.swap_dims(0, 2).sum_dim(1);

        output.into_data().assert_eq(
            &TensorData::new(vec![9.0, 7.0, -1.0, 3.0, 4.0, 5.0], [3, 1, 2]),
            false,
        );
    }

    #[test]
    fn test_should_sum_mid_dim_3d_non_contiguous_2() {
        let tensor = TestTensor::<3>::from([
            [[2.0, 4.0, 1.0], [7.0, -5.0, 3.0]],
            [[3.0, 1.0, 2.0], [4.0, 2.0, 3.0]],
        ]);

        let output = tensor.swap_dims(0, 1).sum_dim(1);

        output.into_data().assert_eq(
            &TensorData::new(vec![5.0, 5.0, 3.0, 11.0, -3.0, 6.0], [2, 1, 3]),
            false,
        );
    }

    #[test]
    fn test_prod_float() {
        let tensor = TestTensor::<2>::from([[2.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let output = tensor.prod();

        // 2 * 1 * 2 * 3 * 4 * 5 = 240 but we need to check the precision because of the float
        let expected = TensorData::from([240.0]);
        output.into_data().assert_approx_eq(&expected, 3);

        let tensor_with_zero = TestTensor::<2>::from([[2.0, 0.0, 2.0], [3.0, 4.0, 5.0]]);
        let output = tensor_with_zero.prod();

        output
            .into_data()
            .assert_eq(&TensorData::from([0.0]), false);
    }

    #[test]
    #[ignore = "Not implemented for all backends yet"]
    fn test_prod_int() {
        let tensor = TestTensorInt::<2>::from([[2, 1, 2], [3, 4, 5]]);
        let output = tensor.prod();

        output
            .into_data()
            .assert_eq(&TensorData::from([240]), false);

        let tensor_with_zero = TestTensorInt::<2>::from([[2, 0, 2], [3, 4, 5]]);
        let output = tensor_with_zero.prod();

        output.into_data().assert_eq(&TensorData::from([0]), false);
    }

    #[test]
    fn test_prod_dim_float() {
        let tensor = TestTensor::<2>::from([[2.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let output = tensor.prod_dim(1);
        let expected = TensorData::from([[4.0], [60.0]]);

        output.into_data().assert_approx_eq(&expected, 4);

        let tensor_with_zero = TestTensor::<2>::from([[2.0, 0.0, 2.0], [3.0, 4.0, 5.0]]);
        let output = tensor_with_zero.prod_dim(1);
        let expected = TensorData::from([[0.0], [60.0]]);

        output.into_data().assert_approx_eq(&expected, 4);
    }

    #[test]
    #[ignore = "Not implemented for all backends yet"]
    fn test_prod_dim_int() {
        let tensor = TestTensorInt::<2>::from([[2, 1, 2], [3, 4, 5]]);
        let output = tensor.prod_dim(1);

        output
            .into_data()
            .assert_eq(&TensorData::from([[4], [60]]), false);

        let tensor_with_zero = TestTensorInt::<2>::from([[2, 0, 2], [3, 4, 5]]);
        let output = tensor_with_zero.prod_dim(1);

        output
            .into_data()
            .assert_eq(&TensorData::from([[0], [60]]), false);
    }
}
