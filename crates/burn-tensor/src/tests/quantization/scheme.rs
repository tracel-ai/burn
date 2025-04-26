#[burn_tensor_testgen::testgen(scheme)]
mod tests {
    use super::*;
    use burn_tensor::{
        Tensor, TensorData,
        quantization::{CalibrationRange, QuantizationMode, QuantizationScheme, QuantizationType},
    };
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

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
            .assert_approx_eq::<FT>(&TensorData::from([0.014_173_228]), Tolerance::default());
        assert!(qparams.offset.is_none());
    }
}
