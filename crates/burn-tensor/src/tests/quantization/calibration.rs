#[burn_tensor_testgen::testgen(calibration)]
mod tests {
    use super::*;
    use burn_tensor::{
        quantization::{
            Calibration, MinMaxCalibration, QuantizationScheme, QuantizationStrategy,
            QuantizationType,
        },
        Tensor,
    };

    #[test]
    fn min_max_calibration_per_tensor_affine_int8() {
        let tensor =
            Tensor::<TestBackend, 1>::from_floats([-1.8, -1.0, 0.0, 0.5], &Default::default());
        let calibration = MinMaxCalibration {
            scheme: QuantizationScheme::PerTensorAffine(QuantizationType::QInt8),
        };

        let strategy = calibration.configure(&tensor);

        if let QuantizationStrategy::PerTensorAffineInt8(q) = strategy {
            assert_eq!(q.scale, 0.009_019_608);
            assert_eq!(q.offset, 72);
        } else {
            panic!("Wrong quantization strategy");
        }
    }

    #[test]
    fn min_max_calibration_per_tensor_symmetric_int8() {
        let tensor =
            Tensor::<TestBackend, 1>::from_floats([-1.8, -1.0, 0.0, 0.5], &Default::default());
        let calibration = MinMaxCalibration {
            scheme: QuantizationScheme::PerTensorSymmetric(QuantizationType::QInt8),
        };

        let strategy = calibration.configure(&tensor);

        if let QuantizationStrategy::PerTensorSymmetricInt8(q) = strategy {
            assert_eq!(q.scale, 0.014_173_228);
        } else {
            panic!("Wrong quantization strategy");
        }
    }
}
