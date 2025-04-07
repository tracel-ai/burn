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
    fn should_support_reshape_maybe_fused_1() {
        let tensor = TestTensorInt::arange(0..32, &Default::default());
        let tensor0 = TestTensorInt::zeros([8, 4, 8], &Default::default());
        let tensor1 = tensor.clone().reshape([1, 4, 8]);
        let output = tensor0 + tensor1;

        let expected = TensorData::from([
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29, 30, 31],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29, 30, 31],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29, 30, 31],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29, 30, 31],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29, 30, 31],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29, 30, 31],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29, 30, 31],
            ],
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29, 30, 31],
            ],
        ]);
        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_reshape_maybe_fused_2() {
        let tensor = TestTensorInt::<3>::from_data([[[0, 2], [1, 2]]], &Default::default());
        let tensor1 = tensor.reshape([2, 2, 1]);
        let tensor2 = TestTensorInt::<3>::full([2, 2, 4], 4, &Default::default());
        let output = tensor2 + tensor1;

        let expected_tensor1 =
            TensorData::from([[[4, 4, 4, 4], [6, 6, 6, 6]], [[5, 5, 5, 5], [6, 6, 6, 6]]]);
        output.into_data().assert_eq(&expected_tensor1, false);
    }

    #[test]
    fn should_support_reshape_maybe_fused_3() {
        let tensor = TestTensorInt::<3>::from_data([[[0, 2], [1, 2]]], &Default::default());
        let tensor1 = tensor.reshape([2, 2, 1]);
        let tensor2 = TestTensorInt::<3>::full([2, 2, 3], 5, &Default::default());

        let expected_tensor1 = TensorData::from([[[0], [2]], [[1], [2]]]);
        tensor1.into_data().assert_eq(&expected_tensor1, false);
    }

    #[test]
    fn should_support_reshape_maybe_fused_4() {
        let tensor = TestTensorInt::<3>::from_data([[[0, 2], [1, 2]]], &Default::default());
        let tensor2 = TestTensorInt::<3>::full([2, 2, 4], 4, &Default::default());
        let tensor2 = tensor2.swap_dims(0, 1);
        let tensor1 = tensor.reshape([2, 2, 1]);
        let output = tensor2 + tensor1;

        let expected_tensor1 =
            TensorData::from([[[4, 4, 4, 4], [6, 6, 6, 6]], [[5, 5, 5, 5], [6, 6, 6, 6]]]);
        output.into_data().assert_eq(&expected_tensor1, false);
    }

    #[test]
    fn should_support_reshape_maybe_fused_5() {
        let tensor = TestTensorInt::<3>::from_data([[[0], [1], [2], [3]]], &Default::default());
        let tensor1 = tensor.clone().reshape([2, 1, 2]);
        let tensor2 = TestTensorInt::<3>::full([2, 4, 2], 0, &Default::default());
        let output = tensor2.clone() + tensor1 + tensor.clone();

        let expected_tensor1 = TensorData::from([
            [[0, 1], [1, 2], [2, 3], [3, 4]],
            [[2, 3], [3, 4], [4, 5], [5, 6]],
        ]);
        output.into_data().assert_eq(&expected_tensor1, false);
    }

    #[test]
    fn should_support_reshape_maybe_fused_6() {
        let device = Default::default();

        let tensor1 = TestTensorInt::arange(0..32, &device);
        let tensor1 = tensor1.reshape([2, 4, 4]);

        let tensor2 = TestTensorInt::arange(0..16, &device);
        let tensor2 = tensor2.reshape([1, 4, 4]);

        let tensor3 = TestTensorInt::arange(0..8, &device);
        let tensor3 = tensor3.reshape([4, 1, 2]);
        let tensor3 = tensor3.swap_dims(0, 2);

        let out = tensor1 + tensor2 + tensor3;

        let expected = TensorData::from([
            [
                [0, 4, 8, 12],
                [8, 12, 16, 20],
                [16, 20, 24, 28],
                [24, 28, 32, 36],
            ],
            [
                [17, 21, 25, 29],
                [25, 29, 33, 37],
                [33, 37, 41, 45],
                [41, 45, 49, 53],
            ],
        ]);
        out.to_data().assert_eq(&expected, false);
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

        output.into_data().assert_eq(&expected, false);
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
