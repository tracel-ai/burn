use super::*;
use burn_tensor::{TensorData, Tolerance};

#[test]
fn should_diff_abs() {
    let data_1 = TensorData::from([[0.0, -1.0], [3.0, 4.0]]);
    let data_2 = TensorData::from([[6.0, 7.0], [9.0, -10.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().abs());
    let tensor_4 = tensor_3.matmul(tensor_2.clone());
    let grads = tensor_4.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([[71.0, 107.0], [71.0, 107.0]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[84.0, 42.0], [90.0, 54.0]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_diff_abs_no_nans() {
    let data_1 = TensorData::from([[6.0, 7.0], [9.0, -10.0]]);
    let data_2 = TensorData::from([[0.0, -1.0], [3.0, 4.0]]);

    let device = AutodiffDevice::new();
    let tensor_1 = TestTensor::<2>::from_data(data_1, &device).require_grad();
    let tensor_2 = TestTensor::from_data(data_2, &device).require_grad();

    let tensor_3 = tensor_1.clone().matmul(tensor_2.clone().abs());
    let grads = tensor_3.backward();

    let grad_1 = tensor_1.grad(&grads).unwrap();
    let grad_2 = tensor_2.grad(&grads).unwrap();

    let expected = TensorData::from([[1.0, 7.0], [1.0, 7.0]]);
    grad_1
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let expected = TensorData::from([[0.0, -15.0], [-3.0, -3.0]]);
    grad_2
        .to_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());

    let contains_nan = grad_2.contains_nan();
    assert!(!contains_nan.into_scalar::<bool>());
}

#[test]
fn should_diff_abs_of_nan() {
    // `abs()`'s backward pass is `grad * sign(input)`. A NaN input must produce
    // a `0` gradient (`sign(NaN) == 0`, matching PyTorch), not a finite value
    // derived from the NaN's incidental sign bit: `sign()` on some backends
    // fell through to `is_positive()` (a sign-*bit* check, not a numeric
    // comparison) for non-zero-equal input, which isn't NaN-aware and so
    // silently turned a NaN gradient into a plausible-looking wrong one.
    let data = TensorData::from([f32::NAN]);

    let device = AutodiffDevice::new();
    let x = TestTensor::<1>::from_data(data, &device).require_grad();

    let y = x.clone().abs();
    let grads = y.backward();
    let grad = x.grad(&grads).unwrap();

    let grad = grad.into_data().convert::<f32>();
    assert_eq!(grad.as_slice::<f32>().unwrap()[0], 0.0);
}
