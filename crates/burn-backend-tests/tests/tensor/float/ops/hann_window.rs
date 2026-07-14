use super::*;
use burn_tensor::signal::hann_window;
use burn_tensor::{DType, TensorData, Tolerance};

#[test]
fn should_support_hann_window_periodic() {
    let tensor: TestTensor<1> = hann_window(8, true, &Default::default());
    let expected = TensorData::from([0.0, 0.146447, 0.5, 0.853553, 1.0, 0.853553, 0.5, 0.146447]);

    // Metal has less precise trigonometric functions.
    let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn should_support_hann_window_symmetric() {
    let tensor: TestTensor<1> = hann_window(8, false, &Default::default());
    let expected = TensorData::from([
        0.0, 0.188255, 0.611260, 0.950484, 0.950484, 0.611260, 0.188255, 0.0,
    ]);

    // Metal has less precise trigonometric functions.
    let tolerance = Tolerance::default().set_half_precision_relative(1e-2);

    tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn should_support_hann_window_options_dtype() {
    let tensor: TestTensor<1> = hann_window(4, true, (&Default::default(), DType::F32));
    assert_eq!(tensor.dtype(), DType::F32);
}

#[test]
fn should_support_hann_window_empty() {
    let tensor: TestTensor<1> = hann_window(0, true, &Default::default());
    assert_eq!(tensor.shape().dims(), [0]);
}

#[test]
fn should_handle_hann_window_size_one_symmetric() {
    let tensor: TestTensor<1> = hann_window(1, false, &Default::default());
    let expected = TensorData::from([1.0]);

    tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_handle_hann_window_size_one_periodic() {
    let tensor: TestTensor<1> = hann_window(1, true, &Default::default());
    let expected = TensorData::from([1.0]);

    tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}
