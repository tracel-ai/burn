use burn_tensor::{
    backend::Backend, AffineQuantization, ElementConversion, Quantization, QuantizationStrategy,
    SymmetricQuantization, Tensor,
};

use super::{QuantizationScheme, QuantizationType};

/// Calibration method used to compute the quantization range mapping.
pub trait Calibration {
    /// Configure the quantization strategy.
    fn configure<B: Backend, const D: usize>(&self, tensor: &Tensor<B, D>) -> QuantizationStrategy;
}

/// Computes the quantization range mapping based on the running min and max values.
pub struct MinMaxCalibration {
    /// Quantization scheme to be used.
    pub scheme: QuantizationScheme,
}

impl Calibration for MinMaxCalibration {
    fn configure<B: Backend, const D: usize>(&self, tensor: &Tensor<B, D>) -> QuantizationStrategy {
        let min = tensor.clone().min().into_scalar().elem::<f32>();
        let max = tensor.clone().max().into_scalar().elem::<f32>();

        match &self.scheme {
            QuantizationScheme::PerTensorAffine(dtype) => match dtype {
                QuantizationType::QInt8 => {
                    QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::new(min, max))
                }
            },
            QuantizationScheme::PerTensorSymmetric(dtype) => match dtype {
                QuantizationType::QInt8 => QuantizationStrategy::PerTensorSymmetricInt8(
                    SymmetricQuantization::new(min, max),
                ),
            },
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::TestBackend;

    #[test]
    fn min_max_calibration_per_tensor_affine_int8() {
        let device = <TestBackend as Backend>::Device::default();
        let tensor = Tensor::<TestBackend, 1>::from_floats([-1.8, -1.0, 0.0, 0.5], &device);
        let calibration = MinMaxCalibration {
            scheme: QuantizationScheme::PerTensorAffine(QuantizationType::QInt8),
        };

        let strategy = calibration.configure(&tensor);

        if let QuantizationStrategy::PerTensorAffineInt8(q) = strategy {
            assert_eq!(q.scale, 0.009019607843137253);
            assert_eq!(q.offset, 72);
        } else {
            panic!("Wrong quantization strategy");
        }
    }

    #[test]
    fn min_max_calibration_per_tensor_symmetric_int8() {
        let device = <TestBackend as Backend>::Device::default();
        let tensor = Tensor::<TestBackend, 1>::from_floats([-1.8, -1.0, 0.0, 0.5], &device);
        let calibration = MinMaxCalibration {
            scheme: QuantizationScheme::PerTensorSymmetric(QuantizationType::QInt8),
        };

        let strategy = calibration.configure(&tensor);

        if let QuantizationStrategy::PerTensorSymmetricInt8(q) = strategy {
            assert_eq!(q.scale, 0.014173228346456693);
        } else {
            panic!("Wrong quantization strategy");
        }
    }
}
