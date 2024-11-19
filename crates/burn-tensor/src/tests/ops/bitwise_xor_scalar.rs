#[burn_tensor_testgen::testgen(bitwise_xor_scalar)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_apply_bitwise_xor_scalar_2d() {
        let tensor_1 = TestTensor::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let scalar = 5.0;

        let output = tensor_1.bitwise_xor_scalar(scalar);

        output
            .into_data()
            .assert_eq(&TensorData::from([[6, 1, 0], [12, 6, 13]]), false);
    }
}
