use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_clamp_ops_int() {
    let tensor = TestTensorInt::<2>::from([[-3, 0, 2], [4, 7, 9]]);

    let output = tensor.clamp(0, 5);

    output
        .into_data()
        .assert_eq(&TensorData::from([[0, 0, 2], [4, 5, 5]]), false);
}

#[test]
fn should_support_clamp_min_max_ops_int() {
    let tensor = TestTensorInt::<2>::from([[-3, 0, 2], [4, 7, 9]]);

    tensor
        .clone()
        .clamp_min(0)
        .into_data()
        .assert_eq(&TensorData::from([[0, 0, 2], [4, 7, 9]]), false);

    tensor
        .clamp_max(5)
        .into_data()
        .assert_eq(&TensorData::from([[-3, 0, 2], [4, 5, 5]]), false);
}
