use super::*;
use burn_tensor::TensorData;

#[test]
fn test_argmax_2d_dim0_int() {
    let tensor = TestTensorInt::<2>::from([[10, 11, 2], [3, 4, 5]]);

    let output = tensor.argmax(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[0, 0, 1]]), false);
}

#[test]
fn test_argmin_2d_dim0_int() {
    let tensor = TestTensorInt::<2>::from([[10, 11, 2], [30, 4, 5]]);

    let output = tensor.argmin(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[0, 1, 0]]), false);
}
