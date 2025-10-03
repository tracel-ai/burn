#[burn_tensor_testgen::testgen(calibration)]
mod tests {
    use super::*;
    use burn_tensor::{
        Tensor, TensorData,
        ops::QuantizedTensor,
        quantization::{Calibration, QTensorPrimitive, QuantLevel, QuantValue, compute_range},
    };

    // NOTE: The scheme variant fields are not important for calibration, only the "main" variant (e.g., per-tensor)
    #[test]
    fn min_max_calibration_range_per_tensor() {
        let tensor = TestTensor::<1>::from_floats([-1.8, -1.0, 0.0, 0.5], &Default::default());
        let scheme = QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S);

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
        let tensor = TestTensor::<2>::from_floats(
            [
                [-1.8, -1.0, 0.0, 0.5],
                [1.8, 1.0, 0.0, -0.5],
                [0.01, 0.02, 0.03, 0.04],
                [-0.01, -0.02, -0.03, -0.04],
            ],
            &Default::default(),
        );
        let scheme = QuantizedTensor::<TestBackend>::default_scheme()
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
}
