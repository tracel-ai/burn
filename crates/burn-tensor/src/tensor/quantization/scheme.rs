use crate::{backend::Backend, Int, Tensor};

use super::{CalibrationRange, QuantizationParameters};

/// Quantization data type.
#[derive(Clone, Debug)]
pub enum QuantizationType {
    /// 8-bit signed integer.
    QInt8,
}

/// Quantization scheme.
#[derive(Clone, Debug)]
pub enum QuantizationScheme {
    /// Per-tensor affine/asymmetric quantization.
    PerTensorAffine(QuantizationType),
    /// Per-tensor symmetric quantization.
    PerTensorSymmetric(QuantizationType),
    // /// Per-channel affine/asymmetric quantization.
    // PerChannelAffine,
    // /// Per-channel symmetric quantization.
    // PerChannelSymmetric,
}

/// Round the tensor to the nearest integer.
fn round<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D, Int> {
    tensor.add_scalar(0.5).int()
}

impl QuantizationScheme {
    /// Compute the quantization parameters.
    pub fn compute_q_params<B: Backend>(
        &self,
        range: CalibrationRange<B>,
    ) -> QuantizationParameters<B> {
        match self {
            QuantizationScheme::PerTensorAffine(dtype) => match dtype {
                QuantizationType::QInt8 => {
                    // Quantized range `[a, b]`
                    let a = i8::min_value() as i32;
                    let b = i8::max_value() as i32;

                    // Input range `[alpha, beta]`
                    let input_range = range.max.clone().sub(range.min.clone());

                    QuantizationParameters {
                        scale: input_range.clone().div_scalar(b - a),
                        offset: Some(round(
                            (range.max.mul_scalar(a) - range.min.mul_scalar(b)).div(input_range),
                        )),
                    }
                }
            },
            QuantizationScheme::PerTensorSymmetric(dtype) => match dtype {
                QuantizationType::QInt8 => {
                    // Quantized range `[a, b]`
                    let b = i8::max_value() as i32;
                    let a = -b;

                    // Compute scale to convert an input value in range `[-alpha, alpha]`
                    let values_range = range.min.abs().max_pair(range.max.abs()).mul_scalar(2);

                    QuantizationParameters {
                        scale: values_range.div_scalar(b - a),
                        offset: None,
                    }
                }
            },
        }
    }
}
