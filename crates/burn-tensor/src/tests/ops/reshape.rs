#[burn_tensor_testgen::testgen(reshape)]
mod tests {
    use super::*;
    use burn_tensor::{Bool, Int, Tensor, TensorData};

    #[test]
    fn should_support_reshape_1d() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = TestTensor::<1>::from_data(data, &Default::default());

        let output = tensor.clone().reshape([1, 3]);
        let expected = TensorData::from([[0.0, 1.0, 2.0]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_reshape_maybe_fused() {
        let tensor = TestTensorInt::arange(0..32, &Default::default());
        let tensor0 = TestTensorInt::zeros([8, 4, 8], &Default::default());
        let tensor1 = tensor.clone().reshape([1, 4, 8]);
        let tensor2 = tensor.reshape([8, 4, 1]);

        let output = tensor0 + tensor1 + tensor2;
        let expected = TensorData::from([
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [9, 10, 11, 12, 13, 14, 15, 16],
                [18, 19, 20, 21, 22, 23, 24, 25],
                [27, 28, 29, 30, 31, 32, 33, 34],
            ],
            [
                [4, 5, 6, 7, 8, 9, 10, 11],
                [13, 14, 15, 16, 17, 18, 19, 20],
                [22, 23, 24, 25, 26, 27, 28, 29],
                [31, 32, 33, 34, 35, 36, 37, 38],
            ],
            [
                [8, 9, 10, 11, 12, 13, 14, 15],
                [17, 18, 19, 20, 21, 22, 23, 24],
                [26, 27, 28, 29, 30, 31, 32, 33],
                [35, 36, 37, 38, 39, 40, 41, 42],
            ],
            [
                [12, 13, 14, 15, 16, 17, 18, 19],
                [21, 22, 23, 24, 25, 26, 27, 28],
                [30, 31, 32, 33, 34, 35, 36, 37],
                [39, 40, 41, 42, 43, 44, 45, 46],
            ],
            [
                [16, 17, 18, 19, 20, 21, 22, 23],
                [25, 26, 27, 28, 29, 30, 31, 32],
                [34, 35, 36, 37, 38, 39, 40, 41],
                [43, 44, 45, 46, 47, 48, 49, 50],
            ],
            [
                [20, 21, 22, 23, 24, 25, 26, 27],
                [29, 30, 31, 32, 33, 34, 35, 36],
                [38, 39, 40, 41, 42, 43, 44, 45],
                [47, 48, 49, 50, 51, 52, 53, 54],
            ],
            [
                [24, 25, 26, 27, 28, 29, 30, 31],
                [33, 34, 35, 36, 37, 38, 39, 40],
                [42, 43, 44, 45, 46, 47, 48, 49],
                [51, 52, 53, 54, 55, 56, 57, 58],
            ],
            [
                [28, 29, 30, 31, 32, 33, 34, 35],
                [37, 38, 39, 40, 41, 42, 43, 44],
                [46, 47, 48, 49, 50, 51, 52, 53],
                [55, 56, 57, 58, 59, 60, 61, 62],
            ],
        ]);
        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_reshape_int() {
        let data = TensorData::from([0, 1, 2]);
        let tensor = TestTensorInt::<1>::from_data(data, &Default::default());

        let output = tensor.clone().reshape([1, 3]);
        let expected = TensorData::from([[0, 1, 2]]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_reshape_bool() {
        let data = TensorData::from([false, true, false]);
        let tensor = TestTensorBool::<1>::from_data(data, &Default::default());

        let output = tensor.clone().reshape([1, 3]);
        let expected = TensorData::from([[false, true, false]]);

        output.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn should_support_reshape_2d() {
        let data = TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]);
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.clone().reshape([6]);
        let expected = TensorData::from([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_dim_infererence() {
        let data = TensorData::from([
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
        ]);
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        // Infer the dimension via -1
        let reshaped = tensor.clone().reshape([2, -1]);
        assert_eq!(reshaped.shape(), [2, 6].into());

        // Infer the dimension via 0 (keep from the source) and -1 (infer)
        let reshaped = reshaped.reshape([0, 2, -1]);
        assert_eq!(reshaped.shape(), [2, 2, 3].into());

        // This is effectively as if we did a flatten
        let reshaped = tensor.clone().reshape([-1]);
        assert_eq!(reshaped.shape(), [12].into());

        // Keeping the first dimension the same (using 0)
        let reshaped = tensor.clone().reshape([0, 3]);
        assert_eq!(reshaped.shape(), [4, 3].into());
    }

    #[test]
    fn should_not_corrupt_after_slice() {
        let zeros = TestTensor::<1>::zeros([2], &Default::default());
        zeros.clone().slice([1..2]).reshape([1]).exp();

        // May lead to zeroes being equal to [0.0, 1.0]
        zeros.into_data().assert_eq(
            &TestTensor::<1>::zeros([2], &Default::default()).to_data(),
            true,
        );
    }

    #[test]
    #[should_panic]
    fn multiple_neg_ones() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = TestTensor::<1>::from_data(data, &Default::default());
        let data_actual = tensor.reshape([-1, -1]).into_data();
    }

    #[test]
    #[should_panic]
    fn neg_value() {
        let data = TensorData::from([0.0, 1.0, 2.0]);
        let tensor = TestTensor::<1>::from_data(data, &Default::default());
        let data_actual = tensor.reshape([-2, -1]).into_data();
    }
}
