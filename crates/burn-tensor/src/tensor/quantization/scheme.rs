#![allow(missing_docs)] // cube derive macros

use serde::{Deserialize, Serialize};

use crate::{Tensor, TensorPrimitive, backend::Backend};

use super::{
    Calibration, CalibrationRange, QuantizationParameters, QuantizationParametersPrimitive,
};

#[cfg(feature = "cubecl")]
use cubecl::prelude::*;

/// Quantization data type.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "cubecl", derive(CubeType, PartialOrd, Ord))]
pub enum QuantizationType {
    /// 8-bit signed integer.
    QInt8,
}

/// Quantization mode.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "cubecl", derive(PartialOrd, Ord))]
pub enum QuantizationMode {
    /// Symmetric or scale quantization.
    Symmetric,
}

/// Quantization scheme.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "cubecl", derive(PartialOrd, Ord))]
pub enum QuantizationScheme {
    /// Per-tensor quantization.
    PerTensor(QuantizationMode, QuantizationType),
}

#[cfg(feature = "cubecl")]
impl CubeType for QuantizationScheme {
    type ExpandType = Self;
}

#[cfg(feature = "cubecl")]
impl CubeDebug for QuantizationScheme {}

#[cfg(feature = "cubecl")]
impl cubecl::frontend::Init for QuantizationScheme {
    fn init(self, _scope: &mut cubecl::ir::Scope, _is_mut: bool) -> Self {
        self
    }
}

impl QuantizationScheme {
    /// Get the [quantization mode](QuantizationMode)
    pub fn mode(&self) -> QuantizationMode {
        match self {
            QuantizationScheme::PerTensor(mode, ..) => *mode,
        }
    }

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
            Calibration::MinMax => match self {
                QuantizationScheme::PerTensor(_, _) => {
                    (B::float_min(tensor.clone()), B::float_max(tensor))
                }
            },
        }
    }

    /// Compute the quantization parameters.
    pub fn compute_q_params<B: Backend>(
        &self,
        range: CalibrationRange<B>,
    ) -> QuantizationParameters<B> {
        match self {
            QuantizationScheme::PerTensor(QuantizationMode::Symmetric, QuantizationType::QInt8) => {
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

    pub fn q_type(&self) -> QuantizationType {
        match self {
            QuantizationScheme::PerTensor(_, quantization_type) => *quantization_type,
        }
    }
}
