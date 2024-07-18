use crate::{backend::Backend, Tensor};

/// The observed input calibration range.
#[derive(Clone, Debug)]
pub struct CalibrationRange<B: Backend> {
    /// Minimum observed value.
    pub min: Tensor<B, 1>,
    /// Maximum observed value.
    pub max: Tensor<B, 1>,
}

/// Calibration method used to compute the quantization range mapping.
pub trait Calibration {
    /// Compute the input tensor range.
    fn compute_range<B: Backend, const D: usize>(
        &self,
        tensor: &Tensor<B, D>,
    ) -> CalibrationRange<B>;
}

/// Computes the per-tensor quantization range mapping based on the min and max values.
pub struct MinMaxCalibration {}

impl Calibration for MinMaxCalibration {
    fn compute_range<B: Backend, const D: usize>(
        &self,
        tensor: &Tensor<B, D>,
    ) -> CalibrationRange<B> {
        let min = tensor.clone().min();
        let max = tensor.clone().max();

        CalibrationRange { min, max }
    }
}

// Observers keep a running min/max, so for static quantization this can be computed multiple times w/ representative data to get the "global" min/max

// pub struct PerChannelCalibrationSettings {
//     pub dtype: QuantizationType,
//     pub symmetric: bool,
// }

// For now, we only support static quantization. Since the tensor is dequantized to a float at the first operation, the remaining operations will all be performed on floats anyways.
// But to test dynamic quantization, just make the first layer use dynamic quantization.

/*
let q_activation = Quantizer {
    calibration: MinMaxCalibration {scheme: QuantizationScheme::PerTensorAffine(QuantizationType::QInt8)},
    dynamic: true,
};
let q_weights = Quantizer {
    calibration: MinMaxCalibration {scheme: QuantizationScheme::PerTensorAffine(QuantizationType::QInt8)},
    dynamic: false,
}

*/
