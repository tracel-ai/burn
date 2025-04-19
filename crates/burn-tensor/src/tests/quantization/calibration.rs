#[burn_tensor_testgen::testgen(calibration)]
mod tests {
    use super::*;
    use burn_tensor::{
        Tensor, TensorData,
        quantization::{Calibration, QuantizationMode, QuantizationScheme, QuantizationType},
    };

    // NOTE: The scheme variant fields are not important for calibration, only the "main" variant (e.g., per-tensor)
    #[test]
    fn min_max_calibration_range_per_tensor() {
        let tensor = TestTensor::<1>::from_floats([-1.8, -1.0, 0.0, 0.5], &Default::default());
        let scheme =
            QuantizationScheme::PerTensor(QuantizationMode::Symmetric, QuantizationType::QInt8);

        let range = scheme.compute_range(&tensor, &Calibration::MinMax);

        range
            .min
            .into_data()
            .assert_eq(&TensorData::from([-1.8]), false);
        range
            .max
            .into_data()
            .assert_eq(&TensorData::from([0.5]), false);
    }
}
