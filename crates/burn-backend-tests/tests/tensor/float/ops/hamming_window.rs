use super::*;
use burn_tensor::signal::hamming_window;
use burn_tensor::{DType, TensorData, Tolerance};

#[test]
fn should_support_hamming_window_periodic() {
    let tensor: TestTensor<1> = hamming_window(8, true, &Default::default());
    let expected = TensorData::from([
        0.086957, 0.220669, 0.543478, 0.866288, 1.0, 0.866288, 0.543478, 0.220669,
    ]);

    // Metal has less precise trigonometric functions.
    let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn should_support_hamming_window_symmetric() {
    let tensor: TestTensor<1> = hamming_window(8, false, &Default::default());
    let expected = TensorData::from([
        0.086957, 0.258842, 0.645064, 0.954790, 0.954790, 0.645064, 0.258842, 0.086957,
    ]);

    // Metal has less precise trigonometric functions.
    let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn should_support_hamming_window_options_dtype() {
    let tensor: TestTensor<1> = hamming_window(4, true, (&Default::default(), DType::F32));
    assert_eq!(tensor.dtype(), DType::F32);
}

#[test]
fn should_support_hamming_window_empty() {
    let tensor: TestTensor<1> = hamming_window(0, true, &Default::default());
    assert_eq!(tensor.shape().dims(), [0]);
}

#[test]
fn should_handle_hamming_window_size_one_symmetric() {
    let tensor: TestTensor<1> = hamming_window(1, false, &Default::default());
    let expected = TensorData::from([1.0]);

    tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_handle_hamming_window_size_one_periodic() {
    let tensor: TestTensor<1> = hamming_window(1, true, &Default::default());
    let expected = TensorData::from([1.0]);

    tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
