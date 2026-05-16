use super::*;
use burn_tensor::{
    TensorData,
    quantization::{Calibration, QuantLevel, QuantValue, compute_range},
};

// NOTE: The scheme variant fields are not important for calibration, only the "main" variant (e.g., per-tensor)
#[test]
fn min_max_calibration_range_per_tensor() {
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([-1.8, -1.0, 0.0, 0.5], &device);
    let scheme = device.default_quant_scheme().with_value(QuantValue::Q8S);

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
        .default_quant_scheme()
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
