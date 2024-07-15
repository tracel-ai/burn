use crate::{backend::Backend, ElementConversion, Tensor};

use super::{
    AffineQuantization, Quantization, QuantizationScheme, QuantizationStrategy, QuantizationType,
    SymmetricQuantization,
};

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
