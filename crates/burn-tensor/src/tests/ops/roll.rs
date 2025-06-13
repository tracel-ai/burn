#[burn_tensor_testgen::testgen(roll)]
mod tests {
    use super::*;
    use burn_tensor::{Int, Slice, Tensor, TensorData, as_type, might_panic, s};

    #[cfg(feature = "std")]
    #[might_panic(reason = "0 size resources are not yet supported")]
    #[test]
    fn test_roll_empty() {
        let device = Default::default();
        let input = TestTensorInt::<2>::zeros([12, 0], &device);

        // Rolling an empty tensor should return the same empty tensor
        input
            .clone()
            .roll(&[0, 1], &[1, 2])
            .to_data()
            .assert_eq(&input.to_data(), false);
    }

    #[test]
    fn test_roll() {
        let input = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

        // No-op shift:
        input
            .clone()
            .roll(&[0, 1], &[0, 0])
            .to_data()
            .assert_eq(&input.clone().to_data(), false);

        input
            .clone()
            .roll(&[0, 1], &[1, -1])
            .to_data()
            .assert_eq(&TensorData::from([[5, 3, 4], [2, 0, 1]]), false);

        input
            .clone()
            .roll(&[0, 1], &[2 * 32 + 1, 3 * (-400) - 1])
            .to_data()
            .assert_eq(&TensorData::from([[5, 3, 4], [2, 0, 1]]), false);
    }

    #[should_panic]
    #[test]
    fn test_roll_dim_too_big() {
        let input = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

        // Attempting to roll on a dimension that doesn't exist should panic
        let _d = input.roll(&[2], &[1]);
    }

    #[should_panic]
    #[test]
    fn test_roll_dim_too_small() {
        let input = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

        // Attempting to roll on a dimension that doesn't exist should panic
        let _d = input.roll(&[-3], &[1]);
    }

    #[should_panic]
    #[test]
    fn test_roll_shift_size_mismatch() {
        let input = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

        // Attempting to roll with a shift size that doesn't match the number of dimensions should panic
        let _d = input.roll(&[0], &[1, 2]);
    }

    #[test]
    fn test_roll_dim() {
        let input = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

        input
            .clone()
            .roll_dim(0, 1)
            .to_data()
            .assert_eq(&TensorData::from([[3, 4, 5], [0, 1, 2]]), false);

        input
            .clone()
            .roll_dim(1, -1)
            .to_data()
            .assert_eq(&TensorData::from([[2, 0, 1], [5, 3, 4]]), false);
    }

    #[should_panic]
    #[test]
    fn test_roll_dim_dim_too_big() {
        let input = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

        // Attempting to roll on a dimension that doesn't exist should panic
        let _d = input.roll_dim(2, 1);
    }

    #[should_panic]
    #[test]
    fn test_roll_dim_dim_too_small() {
        let input = TestTensorInt::<2>::from([[0, 1, 2], [3, 4, 5]]);

        // Attempting to roll on a dimension that doesn't exist should panic
        let _d = input.roll_dim(-3, 1);
    }
}
