use super::*;
use burn_tensor::{ElementConversion, TensorData};

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

// NaN-propagation tests below. Only run under the `flex` backend
// feature; other burn backends follow IEEE 754 min/max and drop NaN.
// Positive-gate form because the default CI build doesn't set
// identifying feature flags on burn-backend-tests. See issue #4814.
#[cfg(feature = "flex")]
#[test]
fn test_cummin_nan_propagation() {
    // Once NaN appears, cummin propagates it forward.
    let tensor = TestTensor::<1>::from([3.0, f32::NAN, 1.0, 2.0]);

    let output = tensor.cummin(0);

    let data: Vec<FloatElem> = output.into_data().to_vec().unwrap();
    assert_eq!(data[0], 3.0.elem::<FloatElem>());
    assert!(data[1].is_nan());
    assert!(data[2].is_nan());
    assert!(data[3].is_nan());
}

#[cfg(feature = "flex")]
#[test]
fn test_cummax_nan_propagation() {
    let tensor = TestTensor::<1>::from([1.0, f32::NAN, 5.0, 2.0]);

    let output = tensor.cummax(0);

    let data: Vec<FloatElem> = output.into_data().to_vec().unwrap();
    assert_eq!(data[0], 1.0.elem::<FloatElem>());
    assert!(data[1].is_nan());
    assert!(data[2].is_nan());
    assert!(data[3].is_nan());
}

#[cfg(feature = "flex")]
#[test]
fn test_cummin_nan_at_start() {
    // NaN on the first element should poison the entire output.
    let tensor = TestTensor::<1>::from([f32::NAN, 1.0, 2.0]);

    let output = tensor.cummin(0);

    let data: Vec<FloatElem> = output.into_data().to_vec().unwrap();
    assert!(data[0].is_nan());
    assert!(data[1].is_nan());
    assert!(data[2].is_nan());
}
