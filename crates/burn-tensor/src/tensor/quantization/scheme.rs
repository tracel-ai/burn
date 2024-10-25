use serde::{Deserialize, Serialize};

use crate::{backend::Backend, Tensor, TensorPrimitive};

use super::{CalibrationRange, QuantizationParameters, QuantizationParametersPrimitive};

/// Quantization data type.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// 8-bit signed integer.
    QInt8,
}

/// Quantization scheme.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
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
                    let a = i8::MIN as i32;
                    let b = i8::MAX as i32;

                    // We extend the `[min, max]` interval to ensure that it contains 0.
                    // Otherwise, we would not meet the requirement that 0 be an exactly
                    // representable value (zero-point).
                    let zero = Tensor::zeros_like(&range.min);
                    let min = range.min.min_pair(zero);
                    let zero = Tensor::zeros_like(&range.max);
                    let max = range.max.max_pair(zero);

                    let scale = max.sub(min.clone()).div_scalar(b - a);
                    let offset = Some(-(min.div(scale.clone()).sub_scalar(a)).int());
                    QuantizationParameters { scale, offset }
                }
            },
            QuantizationScheme::PerTensorSymmetric(dtype) => match dtype {
                QuantizationType::QInt8 => {
                    // Quantized range `[a, b]`
                    let b = i8::MAX as i32;
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

    /// Compute the quantization parameters.
    pub(crate) fn compute_q_params_primitive<B: Backend>(
        &self,
        min: B::FloatTensorPrimitive,
        max: B::FloatTensorPrimitive,
    ) -> QuantizationParametersPrimitive<B> {
        let range = CalibrationRange {
            min: Tensor::from_primitive(TensorPrimitive::Float(min)),
            max: Tensor::from_primitive(TensorPrimitive::Float(max)),
        };
        self.compute_q_params(range).into()
    }
}
