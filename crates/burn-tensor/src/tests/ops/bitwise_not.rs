#[burn_tensor_testgen::testgen(bitwise_not)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_apply_bitwise_not_2d() {
        let tensor_1 = TestTensor::<2>::from([[3, 4, 5], [9, 3, 8]]);

        let output = tensor_1.bitwise_not();

        output
            .into_data()
            .assert_eq(&TensorData::from([[-4, -5, -6], [-10, -4, -9]]), false);
    }
}
