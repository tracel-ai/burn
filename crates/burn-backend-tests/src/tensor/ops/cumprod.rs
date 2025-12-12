use super::*;
use burn_tensor::TensorData;

#[test]
fn test_cumprod_float_dim_0() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let output = tensor.cumprod(0);

    output.into_data().assert_eq(
        &TensorData::from([[1.0, 2.0, 3.0], [4.0, 10.0, 18.0]]),
        false,
    );
}

#[test]
fn test_cumprod_float_dim_1() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let output = tensor.cumprod(1);

    output.into_data().assert_eq(
        &TensorData::from([[1.0, 2.0, 6.0], [4.0, 20.0, 120.0]]),
        false,
    );
}

#[test]
fn test_cumprod_int_dim_0() {
    let tensor = TestTensorInt::<2>::from([[1, 2, 3], [4, 5, 6]]);

    let output = tensor.cumprod(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 2, 3], [4, 10, 18]]), false);
}

#[test]
fn test_cumprod_int_dim_1() {
    let tensor = TestTensorInt::<2>::from([[1, 2, 3], [4, 5, 6]]);

    let output = tensor.cumprod(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 2, 6], [4, 20, 120]]), false);
}

#[test]
fn test_cumprod_float_3d() {
    let tensor = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);

    let output = tensor.cumprod(2);

    output.into_data().assert_eq(
        &TensorData::from([[[1.0, 2.0], [3.0, 12.0]], [[5.0, 30.0], [7.0, 56.0]]]),
        false,
    );
}
