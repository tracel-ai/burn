#![allow(missing_docs)] // cube derive macros

use serde::{Deserialize, Serialize};

use crate::{Shape, Tensor, TensorMetadata, TensorPrimitive, backend::Backend};

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
    /// Get the [quantization mode](QuantizationMode)
    pub fn mode(&self) -> QuantizationMode {
        match self {
            QuantizationScheme::PerTensor(mode, ..) | QuantizationScheme::PerBlock(mode, ..) => {
                *mode
            }
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
                QuantizationScheme::PerBlock(.., layout) => match layout {
                    // For per-block quantization, we can compute the (min, max) range with pooling
                    BlockLayout::Flat(block_size) => {
                        let block_size = *block_size as usize;
                        // Tensor shape must be divisible by block size
                        let shape = tensor.shape();
                        let numel = shape.num_elements();
                        assert_eq!(
                            numel % block_size,
                            0,
                            "Cannot compute per-block quantization range with block size {block_size} and tensor of shape {shape:?}"
                        );
                        let num_blocks = numel / block_size;

                        let tensor = B::float_reshape(tensor, Shape::new([num_blocks, block_size]));
                        let min = B::float_reshape(
                            B::float_min_dim(tensor.clone(), 1),
                            Shape::new([num_blocks]),
                        );
                        let max =
                            B::float_reshape(B::float_max_dim(tensor, 1), Shape::new([num_blocks]));
                        // Tensors with shape [b * num_blocks]
                        (min, max)
                    }
                    BlockLayout::Grid(m, n) => {
                        let (m, n) = (*m as usize, *n as usize);
                        let shape = tensor.shape();
                        let (b, h, w) = match shape.num_dims() {
                            2 => {
                                let [h, w] = shape.dims();
                                (1, h, w)
                            }
                            3 => {
                                let [b, h, w] = shape.dims(); // leading batch dim
                                (b, h, w)
                            }
                            _ => unimplemented!(
                                "Per-block grid quantization is only supported for 2D or 3D tensors"
                            ),
                        };
                        // For optimized dynamic quantization, we probably want a custom kernel that computes the
                        // (min, max) range to quantize each block on-the-fly.
                        // For static quantization, it doesn't really matter.
                        assert!(
                            h % m == 0 && w % n == 0,
                            "Cannot compute per-block quantization range with block grid [{m}, {n}] and tensor of shape {shape:?}"
                        );
                        let num_blocks_h = h / m;
                        let num_blocks_w = w / n;

                        // Max and min pooling
                        let reshaped = B::float_reshape(tensor, Shape::new([b, 1, h, w]));
                        let max = B::max_pool2d(reshaped.clone(), [m, n], [m, n], [0, 0], [1, 1]);
                        let min = B::float_neg(B::max_pool2d(
                            B::float_neg(reshaped),
                            [m, n],
                            [m, n],
                            [0, 0],
                            [0, 0],
                        ));

                        // Tensors with shape [b * num_blocks_h * num_blocks_w]
                        let out_shape = Shape::new([b * num_blocks_h * num_blocks_w]);
                        (
                            B::float_reshape(min, out_shape.clone()),
                            B::float_reshape(max, out_shape),
                        )
                    }
                },
            },
        }
    }

    /// Compute the quantization parameters.
    pub fn compute_q_params<B: Backend>(
        &self,
        range: CalibrationRange<B>,
    ) -> QuantizationParameters<B> {
        // Quantization parameters are computed element-wise based on the calibration range,
        // so it's the same operations for per-tensor and per-block (just that the latter has
        // more parameters)
        match self {
            QuantizationScheme::PerTensor(QuantizationMode::Affine, QuantizationType::QInt8)
            | QuantizationScheme::PerBlock(QuantizationMode::Affine, QuantizationType::QInt8, ..) =>
            {
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
            QuantizationScheme::PerTensor(QuantizationMode::Symmetric, QuantizationType::QInt8)
            | QuantizationScheme::PerBlock(
                QuantizationMode::Symmetric,
                QuantizationType::QInt8,
                ..,
            ) => {
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
            QuantizationScheme::PerBlock(_, quantization_type, _) => *quantization_type,
        }
    }
}
