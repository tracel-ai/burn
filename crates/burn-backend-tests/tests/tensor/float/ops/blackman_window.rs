use super::*;
use burn_tensor::signal::blackman_window;
use burn_tensor::{DType, TensorData, Tolerance};

#[test]
fn should_support_blackman_window_options_dtype() {
    let tensor: TestTensor<1> = blackman_window(4, true, (&Default::default(), DType::F32));
    assert_eq!(tensor.dtype(), DType::F32);
}

#[test]
fn should_support_blackman_window_size_0_symmetric() {
    let tensor: TestTensor<1> = blackman_window(0, false, &Default::default());
    assert_eq!(tensor.dims(), [0]);
}

#[test]
fn should_support_blackman_window_size_0_periodic() {
    let tensor: TestTensor<1> = blackman_window(0, true, &Default::default());
    assert_eq!(tensor.dims(), [0]);
}

#[test]
fn should_handle_blackman_window_size_1_symmetric() {
    let tensor: TestTensor<1> = blackman_window(1, false, &Default::default());
    let expected = TensorData::from([1.0]);

    tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn should_handle_blackman_window_size_1_periodic() {
    let tensor: TestTensor<1> = blackman_window(1, true, &Default::default());
    let expected = TensorData::from([1.0]);

    tensor.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_blackman_window_size_8_periodic() {
    let tensor: TestTensor<1> = blackman_window(8, true, &Default::default());
    let expected = TensorData::from([
        0.0, 6.6447e-02, 3.4000e-01, 7.7355e-01, 1.0000e+00, 7.7355e-01, 3.4000e-01, 6.6447e-02,
    ]);

    // Positions 0 and 7 have values close to 0, hence setting absolute tolerance
    let tolerance = Tolerance::default().set_half_precision_absolute(1e-3);
    tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}

#[test]
fn should_support_blackman_window_size_8_symmetric() {
    let tensor: TestTensor<1> = blackman_window(8, false, &Default::default());
    let expected = TensorData::from([
        0.0, 9.0453e-02, 4.5918e-01, 9.2036e-01, 9.2036e-01, 4.5918e-01, 9.0453e-02, 0.0,
    ]);

    // Positions 0 and 7 have values close to 0, hence setting absolute tolerance
    let tolerance = Tolerance::default().set_half_precision_absolute(1e-3);
    tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, tolerance);
}
