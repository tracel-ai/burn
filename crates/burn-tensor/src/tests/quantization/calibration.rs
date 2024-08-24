#[burn_tensor_testgen::testgen(calibration)]
mod tests {
    use super::*;
    use burn_tensor::{
        quantization::{Calibration, MinMaxCalibration, QuantizationType},
        Tensor, TensorData,
    };

    #[test]
    fn min_max_calibration_range() {
        let tensor =
            Tensor::<TestBackend, 1>::from_floats([-1.8, -1.0, 0.0, 0.5], &Default::default());
        let calibration = MinMaxCalibration {};

        let range = calibration.compute_range(&tensor);

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
