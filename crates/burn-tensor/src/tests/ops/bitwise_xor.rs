#[burn_tensor_testgen::testgen(bitwise_xor)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_apply_bitwise_xor_2d() {
        let tensor_1 = TestTensor::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let tensor_2 = TestTensor::from([[6, 7, 8], [9, 10, 15]]);

        let output = tensor_1.bitwise_xor(tensor_2);

        output
            .into_data()
            .assert_eq(&TensorData::from([[5, 3, 13], [0, 9, 7]]), false);
    }

    #[test]
    fn should_apply_bitwise_xor_1d() {
        let tensor_1 = TestTensor::<1>::from([13, 7]);
        let tensor_2 = TestTensor::from([11, 3]);

        let output = tensor_1.bitwise_xor(tensor_2);

        output
            .into_data()
            .assert_eq(&TensorData::from([6, 4]), false);
    }
}
