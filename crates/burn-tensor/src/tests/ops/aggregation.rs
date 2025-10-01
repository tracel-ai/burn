#[burn_tensor_testgen::testgen(aggregation)]
mod tests {
    use super::*;
    use burn_tensor::backend::Backend;
    use burn_tensor::{Shape, Tensor, TensorData};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_should_mean() {
        let tensor = TestTensor::<2>::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);

        let output = tensor.mean();
        let expected = TensorData::from([15.0 / 6.0]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
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
    fn test_should_sum_dim_maybe_fused() {
        let tensor = TestTensor::<2>::from([[5.0], [-12.0]]);
        let tensor1 = TestTensor::<2>::from([[2.0, 3.0], [-1.0, -5.0]]);
        let ones = TestTensor::<2>::ones([2, 2], &Default::default());
        let x = ones.clone() * tensor;
        let y = ones * tensor1;

        let output = y.clone().sum_dim(1);
        output
            .into_data()
            .assert_eq(&TensorData::from([[5.0], [-6.0]]), false);

        // Negative Indexing.
        let output = y.clone().sum_dim(-1);
        output
            .into_data()
            .assert_eq(&TensorData::from([[5.0], [-6.0]]), false);
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

        let output = tensor.clone().mean_dim(1);
        let expected = TensorData::from([[3.0 / 3.0], [12.0 / 3.0]]);
        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        // Negative Indexing.
        let output = tensor.clone().mean_dim(-1);
        let expected = TensorData::from([[3.0 / 3.0], [12.0 / 3.0]]);
        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
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
        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

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

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        let tensor_with_zero = TestTensor::<2>::from([[2.0, 0.0, 2.0], [3.0, 4.0, 5.0]]);
        let output = tensor_with_zero.prod_dim(1);
        let expected = TensorData::from([[0.0], [60.0]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
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

        // Negative Indexing.
        let tensor_with_zero = TestTensorInt::<2>::from([[2, 0, 2], [3, 4, 5]]);
        let output = tensor_with_zero.prod_dim(-1);
        output
            .into_data()
            .assert_eq(&TensorData::from([[0], [60]]), false);
    }

    #[test]
    fn test_sum_dim_2d() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let output = tensor.clone().sum_dim(1);
        let expected = TensorData::from([[3.], [12.]]);

        output.into_data().assert_eq(&expected, false);

        let output = tensor.sum_dim(0);
        let expected = TensorData::from([[3., 5., 7.]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_sum_dims_2d() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        tensor
            .clone()
            .sum_dims(&[1])
            .to_data()
            .assert_eq(&TensorData::from([[3.], [12.]]), false);

        tensor
            .clone()
            .sum_dims(&[-1])
            .to_data()
            .assert_eq(&TensorData::from([[3.], [12.]]), false);

        tensor
            .clone()
            .sum_dims(&[0, 1])
            .to_data()
            .assert_eq(&TensorData::from([[15.]]), false);
    }

    #[test]
    fn test_sum_dim_1_reshape_maybe_fused() {
        let tensor = TestTensorInt::arange(0..9, &Default::default()).float();
        TestBackend::sync(&tensor.device());

        let output = (tensor.reshape([3, 3]) + 2);
        let output = output.sum_dim(1);
        let expected = TensorData::from([[9.0], [18.0], [27.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_sum_dim_1_swap_dims_maybe_fused() {
        let tensor = TestTensorInt::arange(0..9, &Default::default()).float();
        let tensor = tensor.reshape([3, 3]);
        TestBackend::sync(&tensor.device());

        let output = (tensor.swap_dims(0, 1) + 2);
        let output = output.sum_dim(1);
        let expected = TensorData::from([[15.0], [18.0], [21.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_sum_dim_2_reshape_maybe_fused_broadcast() {
        let tensor = TestTensorInt::arange(0..9, &Default::default()).float();
        TestBackend::sync(&tensor.device());

        let output = (tensor.reshape([1, 3, 3]) + 2);
        let output = output.sum_dim(2);
        let expected = TensorData::from([[[9.0], [18.0], [27.0]]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_sum_dim_2_maybe_fused_on_write() {
        let tensor_1 = TestTensorInt::arange(0..8, &Default::default()).float();
        let tensor_2 = TestTensorInt::arange(10..12, &Default::default()).float();
        let tensor_1 = tensor_1.reshape([1, 2, 4]);
        let tensor_2 = tensor_2.reshape([1, 2, 1]);
        TestBackend::sync(&tensor_1.device());

        let output = (tensor_1 + tensor_2.clone()).sum_dim(2) + tensor_2;
        TestBackend::sync(&output.device());
        let expected = TensorData::from([[[56.0], [77.0]]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_sum_dim_3_maybe_fused_on_read_not_contiguous() {
        let tensor_1 = TestTensorInt::arange(0..8, &Default::default()).float();
        let tensor_2 = TestTensorInt::arange(16..24, &Default::default()).float();

        let tensor_1 = tensor_1.reshape([4, 2, 1]);
        let tensor_1 = tensor_1.swap_dims(0, 2);

        let tensor_2 = tensor_2.reshape([1, 4, 2]);
        let tensor_2 = tensor_2.swap_dims(1, 2);
        TestBackend::sync(&tensor_1.device());

        let output = (tensor_1 + tensor_2).sum_dim(2);
        let expected = TensorData::from([[[88.0], [96.0]]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_sum_dim_4_maybe_fused_on_read_not_contiguous_mixed() {
        let tensor_1 = TestTensorInt::arange(0..8, &Default::default()).float();
        let tensor_2 = TestTensorInt::arange(16..24, &Default::default()).float();
        let tensor_3 = TestTensorInt::arange(32..40, &Default::default()).float();

        let tensor_1 = tensor_1.reshape([4, 2, 1]);
        let tensor_3 = tensor_3.reshape([1, 2, 4]);
        let tensor_1 = tensor_1.swap_dims(0, 2);

        let tensor_2 = tensor_2.reshape([1, 4, 2]);
        let tensor_2 = tensor_2.swap_dims(1, 2);
        TestBackend::sync(&tensor_1.device());

        let output = (tensor_3 + tensor_1 + tensor_2).sum_dim(2);
        let expected = TensorData::from([[[222.0], [246.0]]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_sum_dim_5_maybe_fused_on_read_not_contiguous_mixed() {
        let tensor_1 = TestTensorInt::arange(0..8, &Default::default()).float();
        let tensor_2 = TestTensorInt::arange(16..24, &Default::default()).float();
        let tensor_3 = TestTensorInt::arange(32..40, &Default::default()).float();

        let tensor_1 = tensor_1.reshape([4, 2, 1]);
        let tensor_3 = tensor_3.reshape([1, 2, 4]);
        let tensor_1 = tensor_1.swap_dims(0, 2);

        let tensor_2 = tensor_2.reshape([1, 4, 2]);
        let tensor_2 = tensor_2.swap_dims(1, 2);
        TestBackend::sync(&tensor_1.device());

        let output = (tensor_3 + tensor_1 + tensor_2).sum_dim(1);
        let expected = TensorData::from([[[102.0, 112.0, 122.0, 132.0]]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_sum_dim_6_maybe_fused_on_read_not_contiguous_broadcasted() {
        let tensor_1 = TestTensorInt::arange(0..32, &Default::default()).float();
        let tensor_2 = TestTensorInt::arange(0..8, &Default::default()).float();

        let tensor_1 = tensor_1.reshape([4, 2, 2, 2]);
        let tensor_1 = tensor_1.swap_dims(3, 2);
        let tensor_1 = tensor_1.swap_dims(1, 2);

        let tensor_2 = tensor_2.reshape([1, 2, 2, 2]);

        TestBackend::sync(&tensor_1.device());
        let sum = tensor_2.clone().sum_dim(0);
        let sum = sum.sum_dim(1);
        let sum = sum.sum_dim(2);

        TestBackend::sync(&tensor_1.device());

        let tmp = sum.clone() + 2;
        let output = (tensor_1 + tensor_2 + sum).sum_dim(1);
        let expected = TensorData::from([
            [[[29.0, 43.0], [41.0, 55.0]]],
            [[[45.0, 59.0], [57.0, 71.0]]],
            [[[61.0, 75.0], [73.0, 87.0]]],
            [[[77.0, 91.0], [89.0, 103.0]]],
        ]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_sum_dim_7_maybe_fused_on_read_reshaped() {
        let tensor_1 = TestTensorInt::arange(0..16, &Default::default()).float();

        let tensor_1 = tensor_1.reshape([4, 4]);

        TestBackend::sync(&tensor_1.device());

        let reshaped = tensor_1.reshape([1, 4, 4]);
        let tmp = reshaped + 5.0;
        let output = tmp.sum_dim(2);
        let expected = TensorData::from([[[26.0], [42.0], [58.0], [74.0]]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn test_mean_dim_2d() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        let output = tensor.clone().mean_dim(1);
        let expected = TensorData::from([[1.], [4.]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        let output = tensor.mean_dim(0);
        let expected = TensorData::from([[1.5, 2.5, 3.5]]);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_mean_dims_2d() {
        let tensor =
            TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &Default::default());

        tensor
            .clone()
            .mean_dims(&[1])
            .to_data()
            .assert_eq(&TensorData::from([[1.], [4.]]), false);

        tensor
            .clone()
            .mean_dims(&[-1])
            .to_data()
            .assert_eq(&TensorData::from([[1.], [4.]]), false);

        tensor
            .clone()
            .mean_dims(&[0, 1])
            .to_data()
            .assert_eq(&TensorData::from([[2.5]]), false);
    }
}
