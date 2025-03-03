#[burn_tensor_testgen::testgen(calibration)]
mod tests {
    use super::*;
    use burn_tensor::{
        quantization::{
            BlockLayout, Calibration, QuantizationMode, QuantizationScheme, QuantizationType,
        },
        Tensor, TensorData,
    };

    // NOTE: The scheme variant fields are not important for calibration, only the "main" variant (e.g., per-tensor)
    #[test]
    fn min_max_calibration_range_per_tensor() {
        let tensor = TestTensor::<1>::from_floats([-1.8, -1.0, 0.0, 0.5], &Default::default());
        let scheme =
            QuantizationScheme::PerTensor(QuantizationMode::Affine, QuantizationType::QInt8);

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

    #[test]
    fn min_max_calibration_range_per_block_flat_all() {
        let tensor = TestTensor::<2>::from_floats(
            [[-1.8, -1.0, 0.0, 0.5], [1., 2., 3., 4.]],
            &Default::default(),
        );
        let scheme = QuantizationScheme::PerBlock(
            QuantizationMode::Affine,
            QuantizationType::QInt8,
            BlockLayout::Flat(8),
        );

        let range = scheme.compute_range(&tensor, &Calibration::MinMax);

        range
            .min
            .into_data()
            .assert_eq(&TensorData::from([-1.8]), false);
        range
            .max
            .into_data()
            .assert_eq(&TensorData::from([4.]), false);
    }

    #[test]
    fn min_max_calibration_range_per_block_flat_row() {
        let tensor = TestTensor::<2>::from_floats(
            [[-1.8, -1.0, 0.0, 0.5], [1., 2., 3., 4.]],
            &Default::default(),
        );
        let scheme = QuantizationScheme::PerBlock(
            QuantizationMode::Affine,
            QuantizationType::QInt8,
            BlockLayout::Flat(4),
        );

        let range = scheme.compute_range(&tensor, &Calibration::MinMax);
        range
            .min
            .into_data()
            .assert_eq(&TensorData::from([-1.8, 1.]), false);
        range
            .max
            .into_data()
            .assert_eq(&TensorData::from([0.5, 4.]), false);
    }

    #[test]
    #[should_panic(expected = "Cannot compute per-block quantization range")]
    fn min_max_calibration_range_per_block_flat_invalid() {
        let tensor = TestTensor::<2>::from_floats(
            [[-1.8, -1.0, 0.0, 0.5], [1., 2., 3., 4.]],
            &Default::default(),
        );
        let scheme = QuantizationScheme::PerBlock(
            QuantizationMode::Affine,
            QuantizationType::QInt8,
            BlockLayout::Flat(3),
        );
        let _ = scheme.compute_range(&tensor, &Calibration::MinMax);
    }

    #[test]
    fn min_max_calibration_range_per_block_grid() {
        let tensor = TestTensor::<2>::from_floats(
            [[-1.8, -1.0, 0.0, 0.5], [1., 2., 3., 4.]],
            &Default::default(),
        );
        let scheme = QuantizationScheme::PerBlock(
            QuantizationMode::Affine,
            QuantizationType::QInt8,
            BlockLayout::Grid(2, 2),
        );

        let range = scheme.compute_range(&tensor, &Calibration::MinMax);

        range
            .min
            .into_data()
            .assert_eq(&TensorData::from([-1.8, 0.]), false);
        range
            .max
            .into_data()
            .assert_eq(&TensorData::from([2., 4.]), false);
    }

    #[test]
    #[should_panic(expected = "Cannot compute per-block quantization range")]
    fn min_max_calibration_range_per_block_grid_invalid() {
        let tensor = TestTensor::<2>::from_floats(
            [[-1.8, -1.0, 0.0, 0.5], [1., 2., 3., 4.]],
            &Default::default(),
        );
        let scheme = QuantizationScheme::PerBlock(
            QuantizationMode::Affine,
            QuantizationType::QInt8,
            BlockLayout::Grid(3, 3),
        );

        let _ = scheme.compute_range(&tensor, &Calibration::MinMax);
    }
}
