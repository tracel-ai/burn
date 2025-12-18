use super::*;
use burn_tensor::TensorData;

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
fn test_cumsum_float_3d() {
    let tensor = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);

    let output = tensor.cumsum(2);

    output.into_data().assert_eq(
        &TensorData::from([[[1.0, 3.0], [3.0, 7.0]], [[5.0, 11.0], [7.0, 15.0]]]),
        false,
    );
}

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
fn test_cumprod_float_3d() {
    let tensor = TestTensor::<3>::from([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);

    let output = tensor.cumprod(2);

    output.into_data().assert_eq(
        &TensorData::from([[[1.0, 2.0], [3.0, 12.0]], [[5.0, 30.0], [7.0, 56.0]]]),
        false,
    );
}

#[test]
fn test_cummin_float_dim_0() {
    let tensor = TestTensor::<2>::from([[3.0, 1.0, 4.0], [2.0, 5.0, 1.0]]);

    let output = tensor.cummin(0);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3.0, 1.0, 4.0], [2.0, 1.0, 1.0]]), false);
}

#[test]
fn test_cummin_float_dim_1() {
    let tensor = TestTensor::<2>::from([[3.0, 1.0, 4.0], [2.0, 5.0, 1.0]]);

    let output = tensor.cummin(1);

    output
        .into_data()
        .assert_eq(&TensorData::from([[3.0, 1.0, 1.0], [2.0, 2.0, 1.0]]), false);
}

#[test]
fn test_cummin_float_3d() {
    let tensor = TestTensor::<3>::from([[[4.0, 2.0], [3.0, 1.0]], [[5.0, 6.0], [7.0, 8.0]]]);

    let output = tensor.cummin(2);

    output.into_data().assert_eq(
        &TensorData::from([[[4.0, 2.0], [3.0, 1.0]], [[5.0, 5.0], [7.0, 7.0]]]),
        false,
    );
}

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
fn test_cummax_float_3d() {
    let tensor = TestTensor::<3>::from([[[1.0, 3.0], [2.0, 4.0]], [[5.0, 2.0], [6.0, 1.0]]]);

    let output = tensor.cummax(2);

    output.into_data().assert_eq(
        &TensorData::from([[[1.0, 3.0], [2.0, 4.0]], [[5.0, 5.0], [6.0, 6.0]]]),
        false,
    );
}
