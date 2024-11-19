#[burn_tensor_testgen::testgen(bitwise_or_scalar)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_apply_bitwise_or_scalar_2d() {
        let tensor_1 = TestTensor::<2>::from([[3, 4, 5], [9, 3, 8]]);
        let scalar = 5.0;

        let output = tensor_1.bitwise_or_scalar(scalar);

        output
            .into_data()
            .assert_eq(&TensorData::from([[7, 5, 5], [13, 7, 13]]), false);
    }
}
