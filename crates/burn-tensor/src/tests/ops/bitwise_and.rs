#[burn_tensor_testgen::testgen(bitwise_and)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_apply_bitwise_and_2d() {
        let tensor_1 = TestTensor::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let tensor_2 = TestTensor::from([[6, 7, 8], [9, 10, 15]]);

        let output = tensor_1.bitwise_and(tensor_2);

        output
            .into_data()
            .assert_eq(&TensorData::from([[2, 4, 0], [9, 2, 8]]), false);
    }

    #[test]
    fn should_apply_bitwise_and_1d() {
        let tensor_1 = TestTensor::<1>::from([13, 7]);
        let tensor_2 = TestTensor::from([11, 3]);

        let output = tensor_1.bitwise_and(tensor_2);

        output
            .into_data()
            .assert_eq(&TensorData::from([9, 3]), false);
    }
}
