#![allow(missing_docs)] // cube derive macros

use serde::{Deserialize, Serialize};

use crate::{backend::Backend, Tensor, TensorPrimitive};

use super::{CalibrationRange, QuantizationParameters, QuantizationParametersPrimitive};

#[cfg(feature = "cubecl")]
use cubecl::prelude::*;

/// Quantization data type.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "cubecl", derive(CubeType, PartialOrd, Ord))]
pub enum QuantizationType {
    /// 8-bit signed integer.
    QInt8,
}

// CubeType not implemented for usize
/// Block quantization layout.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "cubecl", derive(CubeType, PartialOrd, Ord))]
pub enum BlockLayout {
    /// The tensor is split into linear segments of N elements.
    Flat(u32),
    /// The tensor is split into segments of M x N elements.
    Grid(u32, u32),
}
/// Quantization mode.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "cubecl", derive(PartialOrd, Ord))]
pub enum QuantizationMode {
    /// Affine or asymmetric quantization.
    Affine,
    /// Symmetric or scale quantization.
    Symmetric,
}

/// Quantization scheme.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "cubecl", derive(PartialOrd, Ord))]
pub enum QuantizationScheme {
    /// Per-tensor quantization.
    PerTensor(QuantizationMode, QuantizationType),
    /// Per-block quantization.
    PerBlock(QuantizationMode, QuantizationType, BlockLayout),
}

#[cfg(feature = "cubecl")]
impl CubeType for QuantizationScheme {
    type ExpandType = Self;
}

#[cfg(feature = "cubecl")]
impl CubeDebug for QuantizationScheme {}

#[cfg(feature = "cubecl")]
impl cubecl::frontend::Init for QuantizationScheme {
    fn init(self, _scope: &mut cubecl::ir::Scope) -> Self {
        self
    }
}

impl QuantizationScheme {
    /// Compute the quantization parameters.
    pub fn compute_q_params<B: Backend>(
        &self,
        range: CalibrationRange<B>,
    ) -> QuantizationParameters<B> {
        match self {
            QuantizationScheme::PerTensor(QuantizationMode::Affine, QuantizationType::QInt8) => {
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

                // If scale is 0 (most likely due to a tensor full of zeros), we arbitrarily adjust the
                // scale to 0.1 to avoid division by zero.
                let scale = max.sub(min.clone()).div_scalar(b - a);
                let scale = scale.clone().mask_fill(scale.equal_elem(0.), 0.1);
                let offset = Some(-(min.div(scale.clone()).sub_scalar(a)).int());
                QuantizationParameters { scale, offset }
            }
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
            QuantizationScheme::PerBlock(_mode, _dtype, _block_layout) => todo!(),
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
