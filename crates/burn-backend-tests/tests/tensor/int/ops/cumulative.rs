use super::*;
use burn_tensor::TensorData;

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
fn test_cummin_int_dim_0() {
    let tensor = TestTensorInt::<2>::from([[3, 1, 4], [2, 5, 1]]);

    let output = tensor.cummin(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3, 1, 4], [2, 1, 1]]), false);
}

#[test]
fn test_cummin_int_dim_1() {
    let tensor = TestTensorInt::<2>::from([[3, 1, 4], [2, 5, 1]]);

    let output = tensor.cummin(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3, 1, 1], [2, 2, 1]]), false);
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
