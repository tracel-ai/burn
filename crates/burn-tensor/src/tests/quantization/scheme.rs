#[burn_tensor_testgen::testgen(scheme)]
mod tests {
    use super::*;
    use burn_tensor::{
        quantization::{
            BlockLayout, CalibrationRange, QuantizationMode, QuantizationScheme, QuantizationType,
        },
        Tensor, TensorData,
    };

    #[test]
    fn per_tensor_affine_int8() {
        let device = Default::default();
        let scheme =
            QuantizationScheme::PerTensor(QuantizationMode::Affine, QuantizationType::QInt8);
        let range = CalibrationRange {
            min: TestTensor::<1>::from_floats([-1.8], &device),
            max: TestTensor::<1>::from_floats([0.5], &device),
        };

        let qparams = scheme.compute_q_params(range);

        qparams
            .scale
            .into_data()
            .assert_approx_eq(&TensorData::from([0.009_019_608]), 8);
        qparams
            .offset
            .unwrap()
            .into_data()
            .assert_eq(&TensorData::from([71]), false);
    }

    #[test]
    fn per_tensor_symmetric_int8() {
        let device = Default::default();
        let scheme =
            QuantizationScheme::PerTensor(QuantizationMode::Symmetric, QuantizationType::QInt8);
        let range = CalibrationRange {
            min: TestTensor::<1>::from_floats([0.5], &device),
            max: TestTensor::<1>::from_floats([1.8], &device),
        };

        let qparams = scheme.compute_q_params(range);

        qparams
            .scale
            .into_data()
            .assert_approx_eq(&TensorData::from([0.014_173_228]), 8);
        assert!(qparams.offset.is_none());
    }

    #[test]
    fn per_block_affine_int8() {
        let device = Default::default();
        let scheme = QuantizationScheme::PerBlock(
            QuantizationMode::Affine,
            QuantizationType::QInt8,
            BlockLayout::Flat(3), // layout doesn't matter when computing qparams
        );
        let range = CalibrationRange {
            min: TestTensor::<1>::from_floats([-1.8, -2.0, 0.5], &device),
            max: TestTensor::<1>::from_floats([0.5, 1.5, 1.8], &device),
        };

        let qparams = scheme.compute_q_params(range);

        qparams.scale.into_data().assert_approx_eq(
            &TensorData::from([0.009_019_608, 0.013_725_490, 0.007_0588_234]),
            8,
        );
        qparams
            .offset
            .unwrap()
            .into_data()
            .assert_eq(&TensorData::from([71, 17, -128]), false);
    }

    #[test]
    fn per_block_symmetric_int8() {
        let device = Default::default();
        let scheme = QuantizationScheme::PerBlock(
            QuantizationMode::Symmetric,
            QuantizationType::QInt8,
            BlockLayout::Flat(3), // layout doesn't matter when computing qparams
        );
        let range = CalibrationRange {
            min: TestTensor::<1>::from_floats([-1.8, -2.0, 0.5], &device),
            max: TestTensor::<1>::from_floats([0.5, 1.5, 1.8], &device),
        };

        let qparams = scheme.compute_q_params(range);

        qparams.scale.into_data().assert_approx_eq(
            &TensorData::from([0.014_173_228, 0.015_748_031, 0.014_173_228]),
            8,
        );
        assert!(qparams.offset.is_none());
    }
}
