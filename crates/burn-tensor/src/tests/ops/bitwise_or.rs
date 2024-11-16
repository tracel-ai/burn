#[burn_tensor_testgen::testgen(bitwise_or)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_apply_bitwise_or_2d() {
        let tensor_1 = TestTensor::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let tensor_2 = TestTensor::from([[6, 7, 8], [9, 10, 15]]);

        let output = tensor_1.bitwise_or(tensor_2);

        output
            .into_data()
            .assert_eq(&TensorData::from([[7, 7, 13], [9, 11, 15]]), false);
    }

    #[test]
    fn should_apply_bitwise_or_1d() {
        let tensor_1 = TestTensor::<1>::from([13, 7]);
        let tensor_2 = TestTensor::from([11, 3]);

        let output = tensor_1.bitwise_or(tensor_2);

        output
            .into_data()
            .assert_eq(&TensorData::from([15, 7]), false);
    }
}
