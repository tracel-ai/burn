#[burn_tensor_testgen::testgen(scheme)]
mod tests {
    use super::*;
    use burn_tensor::{
        quantization::{CalibrationRange, QuantizationMode, QuantizationScheme, QuantizationType},
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
            min: TestTensor::<1>::from_floats([-1.8], &device),
            max: TestTensor::<1>::from_floats([0.5], &device),
        };

        let qparams = scheme.compute_q_params(range);

        qparams
            .scale
            .into_data()
            .assert_approx_eq(&TensorData::from([0.014_173_228]), 8);
        assert!(qparams.offset.is_none());
    }
}
