use crate::*;
use burn_tensor::{ TensorData};

#[test]
fn test_cummax_float_dim_0() {
    let tensor = TestTensor::<2>::from([[3.0, 1.0, 4.0], [1.0, 5.0, 2.0]]);

    let output = tensor.cummax(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3.0, 1.0, 4.0], [3.0, 5.0, 4.0]]), false);
}

#[test]
fn test_cummax_float_dim_1() {
    let tensor = TestTensor::<2>::from([[3.0, 1.0, 4.0], [1.0, 5.0, 2.0]]);

    let output = tensor.cummax(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3.0, 3.0, 4.0], [1.0, 5.0, 5.0]]), false);
}

#[test]
fn test_cummax_int_dim_0() {
    let tensor = TestTensorInt::<2>::from([[3, 1, 4], [1, 5, 2]]);

    let output = tensor.cummax(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3, 1, 4], [3, 5, 4]]), false);
}

#[test]
fn test_cummax_int_dim_1() {
    let tensor = TestTensorInt::<2>::from([[3, 1, 4], [1, 5, 2]]);

    let output = tensor.cummax(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3, 3, 4], [1, 5, 5]]), false);
}

#[test]
fn test_cummax_float_3d() {
    let tensor = TestTensor::<3>::from([[[1.0, 3.0], [2.0, 4.0]], [[5.0, 2.0], [6.0, 1.0]]]);

    let output = tensor.cummax(2);

    output.into_data().assert_eq(
        &TensorData::from([[[1.0, 3.0], [2.0, 4.0]], [[5.0, 5.0], [6.0, 6.0]]]),
        false,
    );
}
