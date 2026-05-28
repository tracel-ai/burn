use super::*;
use burn_tensor::{
    TensorData, Tolerance,
    quantization::{Calibration, QuantLevel, QuantValue, compute_range},
};

// NOTE: The scheme variant fields are not important for calibration, only the "main" variant (e.g., per-tensor)
#[test]
fn min_max_calibration_range_per_tensor() {
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([-1.8, -1.0, 0.0, 0.5], &device);
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);

    let range = compute_range(&scheme, &tensor, &Calibration::MinMax);

    range
        .min
        .into_data()
        .assert_eq(&TensorData::from([-1.8]), false);
    range
        .max
        .into_data()
        .assert_eq(&TensorData::from([0.5]), false);
}

#[test]
fn min_max_calibration_range_per_block() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data(
        [
            [-1.8, -1.0, 0.0, 0.5],
            [1.8, 1.0, 0.0, -0.5],
            [0.01, 0.02, 0.03, 0.04],
            [-0.01, -0.02, -0.03, -0.04],
        ],
        &device,
    );
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::block([4]));

    let range = compute_range(&scheme, &tensor, &Calibration::MinMax);

    range
        .min
        .into_data()
        .assert_eq(&TensorData::from([[-1.8], [-0.5], [0.01], [-0.04]]), false);
    range
        .max
        .into_data()
        .assert_eq(&TensorData::from([[0.5], [1.8], [0.04], [-0.01]]), false);
}

// abs-mean calibration: gamma = mean(|W|), range = [-gamma, +gamma]
// weights: [-0.9, -0.3, 0.0, 0.6]  => mean(|w|) = (0.9+0.3+0.0+0.6)/4 = 0.45
#[test]
fn abs_mean_calibration_range_per_tensor() {
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([-0.9_f32, -0.3, 0.0, 0.6], &device);
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q2S);

    let range = compute_range(&scheme, &tensor, &Calibration::AbsMean);

    range
        .min
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([-0.45_f32]), Tolerance::default());
    range
        .max
        .into_data()
        .assert_approx_eq::<FloatElem>(&TensorData::from([0.45_f32]), Tolerance::default());
}

// block abs-mean: 2 blocks of 4 weights each
// block 0: [-0.9, -0.3, 0.0, 0.6]  gamma = 0.45
// block 1: [0.1, 0.2, 0.3, 0.4]    gamma = 0.25
#[test]
fn abs_mean_calibration_range_per_block() {
    let device = Default::default();
    let tensor =
        TestTensor::<2>::from_data([[-0.9_f32, -0.3, 0.0, 0.6], [0.1, 0.2, 0.3, 0.4]], &device);
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q2S)
        .with_level(QuantLevel::block([4]));

    let range = compute_range(&scheme, &tensor, &Calibration::AbsMean);

    range.min.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[-0.45_f32], [-0.25]]),
        Tolerance::default(),
    );
    range.max.into_data().assert_approx_eq::<FloatElem>(
        &TensorData::from([[0.45_f32], [0.25]]),
        Tolerance::default(),
    );
}
