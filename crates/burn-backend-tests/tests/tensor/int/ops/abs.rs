use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_abs_ops_int() {
    let tensor = TestTensorInt::<2>::from([[0, -1, 2], [3, 4, -5]]);

    let output = tensor.abs();

    output
        .into_data()
        .assert_eq(&TensorData::from([[0, 1, 2], [3, 4, 5]]), false);
}
