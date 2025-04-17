#![allow(missing_docs)] // cube derive macros

use serde::{Deserialize, Serialize};

use crate::{Tensor, TensorPrimitive, backend::Backend};

use super::{
    Calibration, CalibrationRange, QuantizationParameters, QuantizationParametersPrimitive,
};

/// Quantization scheme.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct QuantizationScheme {
    pub level: QuantizationLevel,
    pub mode: QuantizationMode,
    pub q_type: QuantizationType,
    pub acc_precision: QuantizationAccPrecision,
    pub output: QuantizationOutput,
}

impl Default for QuantizationScheme {
    fn default() -> Self {
        Self {
            level: QuantizationLevel::Tensor,
            mode: QuantizationMode::Symmetric,
            q_type: QuantizationType::QInt8,
            acc_precision: QuantizationAccPrecision::Full,
            output: QuantizationOutput::Dequantized,
        }
    }
}

impl QuantizationScheme {
    pub fn set_level(mut self, level: QuantizationLevel) -> Self {
        self.level = level;
        self
    }

    pub fn set_mode(mut self, mode: QuantizationMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn set_q_type(mut self, q_type: QuantizationType) -> Self {
        self.q_type = q_type;
        self
    }

    pub fn set_acc_precision(mut self, acc_precision: QuantizationAccPrecision) -> Self {
        self.acc_precision = acc_precision;
        self
    }

    pub fn set_output(mut self, output: QuantizationOutput) -> Self {
        self.output = output;
        self
    }
}

/// Quantization level.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantizationLevel {
    /// Quantize the whole tensor using a single tensor.
    Tensor,
}

/// Quantization data type.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantizationType {
    /// 8-bit signed integer.
    QInt8,
}

/// Quantization mode.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantizationMode {
    /// Symmetric or scale quantization.
    Symmetric,
}

/// Quantization accumulator precision. This is the precision to used when accumulating values
/// while executing algorithms such as matmul.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantizationAccPrecision {
    Full,
    Half,
}

/// Quantization accumulator precision. This is the precision to used when accumulating values
/// while executing algorithms such as matmul.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantizationOutput {
    Quantized,
    Dequantized,
}

impl QuantizationScheme {
    /// Compute the quantization range mapping.
    pub fn compute_range<B: Backend, const D: usize>(
        &self,
        tensor: &Tensor<B, D>,
        calibration: &Calibration,
    ) -> CalibrationRange<B> {
        let (min, max) = match &tensor.primitive {
            TensorPrimitive::Float(tensor) => {
                self.compute_range_primitive::<B>(tensor.clone(), calibration)
            }
            TensorPrimitive::QFloat(_) => unreachable!(),
        };

        CalibrationRange {
            min: Tensor::from_primitive(TensorPrimitive::Float(min)),
            max: Tensor::from_primitive(TensorPrimitive::Float(max)),
        }
    }

    pub(crate) fn compute_range_primitive<B: Backend>(
        &self,
        tensor: B::FloatTensorPrimitive,
        calibration: &Calibration,
    ) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
        match calibration {
            Calibration::MinMax => match self.level {
                QuantizationLevel::Tensor => (B::float_min(tensor.clone()), B::float_max(tensor)),
            },
        }
    }

    /// Compute the quantization parameters.
    pub fn compute_q_params<B: Backend>(
        &self,
        range: CalibrationRange<B>,
    ) -> QuantizationParameters<B> {
        match self {
            QuantizationScheme {
                level: QuantizationLevel::Tensor,
                mode: QuantizationMode::Symmetric,
                q_type: QuantizationType::QInt8,
                acc_precision: _,
                output: _,
            } => {
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
