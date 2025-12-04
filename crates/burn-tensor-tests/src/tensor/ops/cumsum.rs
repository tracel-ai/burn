use crate::*;
use burn_tensor::{ TensorData};

#[test]
fn test_cumsum_float_dim_0() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let output = tensor.cumsum(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1.0, 2.0, 3.0], [5.0, 7.0, 9.0]]), false);
}

#[test]
fn test_cumsum_float_dim_1() {
    let tensor = TestTensor::<2>::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let output = tensor.cumsum(1);

    output.into_data().assert_eq(
        &TensorData::from([[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]]),
        false,
    );
}

#[test]
fn test_cumsum_non_contiguous() {
    let tensor = TestTensor::<2>::from([[1., 2.], [3., 4.]]).swap_dims(0, 1);

    let output = tensor.cumsum(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1., 4.], [2., 6.]]), false);
}

#[test]
fn test_cumsum_int_dim_0() {
    let tensor = TestTensorInt::<2>::from([[1, 2, 3], [4, 5, 6]]);

    let output = tensor.cumsum(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 2, 3], [5, 7, 9]]), false);
}

#[test]
fn test_cumsum_int_dim_1() {
    let tensor = TestTensorInt::<2>::from([[1, 2, 3], [4, 5, 6]]);

    let output = tensor.cumsum(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1, 3, 6], [4, 9, 15]]), false);
}

#[test]
fn test_cumsum_float_3d() {
    let tensor = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);

    let output = tensor.cumsum(2);

    output.into_data().assert_eq(
        &TensorData::from([[[1.0, 3.0], [3.0, 7.0]], [[5.0, 11.0], [7.0, 15.0]]]),
        false,
    );
}
