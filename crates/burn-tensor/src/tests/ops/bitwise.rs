#[burn_tensor_testgen::testgen(bitwise)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_apply_bitwise_and_2d() {
        let tensor_1 = TestTensorInt::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let tensor_2 = TestTensorInt::from([[6, 7, 8], [9, 10, 15]]);

        let output = tensor_1.bitwise_and(tensor_2);

        output
            .into_data()
            .assert_eq(&TensorData::from([[2, 4, 0], [9, 2, 8]]), false);
    }

    #[test]
    fn should_apply_bitwise_and_1d() {
        let tensor_1 = TestTensorInt::<1>::from([13, 7]);
        let tensor_2 = TestTensorInt::from([11, 3]);

        let output = tensor_1.bitwise_and(tensor_2);

        output
            .into_data()
            .assert_eq(&TensorData::from([9, 3]), false);
    }

    #[test]
    fn should_apply_bitwise_and_scalar_2d() {
        let tensor_1 = TestTensorInt::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let scalar = 5;

        let output = tensor_1.bitwise_and_scalar(scalar);

        output
            .into_data()
            .assert_eq(&TensorData::from([[1, 4, 5], [1, 1, 0]]), false);
    }

    #[test]
    fn should_apply_bitwise_not_2d() {
        let tensor_1 = TestTensorInt::<2>::from([[3, 4, 5], [9, 3, 8]]);

        let output = tensor_1.bitwise_not();

        output
            .into_data()
            .assert_eq(&TensorData::from([[-4, -5, -6], [-10, -4, -9]]), false);
    }

    #[test]
    fn should_apply_bitwise_or_scalar_2d() {
        let tensor_1 = TestTensorInt::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let scalar = 5;

        let output = tensor_1.bitwise_or_scalar(scalar);

        output
            .into_data()
            .assert_eq(&TensorData::from([[7, 5, 5], [13, 7, 13]]), false);
    }

    #[test]
    fn should_apply_bitwise_or_2d() {
        let tensor_1 = TestTensorInt::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let tensor_2 = TestTensorInt::from([[6, 7, 8], [9, 10, 15]]);

        let output = tensor_1.bitwise_or(tensor_2);

        output
            .into_data()
            .assert_eq(&TensorData::from([[7, 7, 13], [9, 11, 15]]), false);
    }

    #[test]
    fn should_apply_bitwise_or_1d() {
        let tensor_1 = TestTensorInt::<1>::from([13, 7]);
        let tensor_2 = TestTensorInt::from([11, 3]);

        let output = tensor_1.bitwise_or(tensor_2);

        output
            .into_data()
            .assert_eq(&TensorData::from([15, 7]), false);
    }

    #[test]
    fn should_apply_bitwise_xor_scalar_2d() {
        let tensor_1 = TestTensorInt::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let scalar = 5;

        let output = tensor_1.bitwise_xor_scalar(scalar);

        output
            .into_data()
            .assert_eq(&TensorData::from([[6, 1, 0], [12, 6, 13]]), false);
    }

    #[test]
    fn should_apply_bitwise_xor_2d() {
        let tensor_1 = TestTensorInt::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let tensor_2 = TestTensorInt::from([[6, 7, 8], [9, 10, 15]]);

        let output = tensor_1.bitwise_xor(tensor_2);

        output
            .into_data()
            .assert_eq(&TensorData::from([[5, 3, 13], [0, 9, 7]]), false);
    }

    #[test]
    fn should_apply_bitwise_xor_1d() {
        let tensor_1 = TestTensorInt::<1>::from([13, 7]);
        let tensor_2 = TestTensorInt::from([11, 3]);

        let output = tensor_1.bitwise_xor(tensor_2);

        output
            .into_data()
            .assert_eq(&TensorData::from([6, 4]), false);
    }

    #[test]
    fn should_apply_bitwise_left_shift_2d() {
        let tensor_1 = TestTensorInt::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let tensor_2 = TestTensorInt::from([[1, 2, 3], [4, 5, 6]]);

        let output = tensor_1.bitwise_left_shift(tensor_2);

        output
            .into_data()
            .assert_eq(&TensorData::from([[6, 16, 40], [144, 96, 512]]), false);
    }

    #[test]
    fn should_apply_bitwise_left_shift_scalar_2d() {
        let tensor_1 = TestTensorInt::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let scalar = 2;

        let output = tensor_1.bitwise_left_shift_scalar(scalar);

        output
            .into_data()
            .assert_eq(&TensorData::from([[12, 16, 20], [36, 12, 32]]), false);
    }

    #[test]
    fn should_apply_bitwise_right_shift_2d() {
        let tensor_1 = TestTensorInt::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let tensor_2 = TestTensorInt::from([[1, 2, 3], [4, 5, 6]]);

        let output = tensor_1.bitwise_right_shift(tensor_2);

        output
            .into_data()
            .assert_eq(&TensorData::from([[1, 1, 0], [0, 0, 0]]), false);
    }

    #[test]
    fn should_apply_bitwise_right_shift_scalar_2d() {
        let tensor_1 = TestTensorInt::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let scalar = 2;

        let output = tensor_1.bitwise_right_shift_scalar(scalar);

        output
            .into_data()
            .assert_eq(&TensorData::from([[0, 1, 1], [2, 0, 2]]), false);
    }
}
