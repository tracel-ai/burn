#[burn_tensor_testgen::testgen(bitwise_and_scalar)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_apply_bitwise_and_scalar_2d() {
        let tensor_1 = TestTensor::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let scalar = 5.0;

        let output = tensor_1.bitwise_and_scalar(scalar);

        output
            .into_data()
            .assert_eq(&TensorData::from([[1, 4, 5], [1, 1, 0]]), false);
    }
}
