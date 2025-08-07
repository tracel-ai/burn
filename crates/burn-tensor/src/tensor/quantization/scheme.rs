// We re-export those types.
pub use cubecl_quant::scheme::{
    QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};

use serde::{Deserialize, Serialize};

use crate::{Shape, Tensor, TensorMetadata, TensorPrimitive, backend::Backend};

use super::{
    Calibration, CalibrationRange, QuantizationParameters, QuantizationParametersPrimitive,
};

/// Describes a quantization scheme/configuration.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct QuantSettings {
    /// The scheme used for quantization.
    pub scheme: QuantScheme,
    /// Precision used for accumulating intermediate values (e.g., during matmul).
    pub acc_precision: QuantAcc,
    /// Whether to propagate quantization to outputs or return unquantized results.
    pub propagation: QuantPropagation,
}

impl Default for QuantSettings {
    fn default() -> Self {
        Self {
            scheme: Default::default(),
            acc_precision: QuantAcc::F32,
            propagation: QuantPropagation::Inhibit,
        }
    }
}

impl QuantSettings {
    /// Set the quantization scheme.
    pub fn with_scheme(mut self, scheme: QuantScheme) -> Self {
        self.scheme = scheme;
        self
    }
    /// Set the accumulation precision used during computations.
    pub fn with_acc_precision(mut self, acc_precision: QuantAcc) -> Self {
        self.acc_precision = acc_precision;
        self
    }

    /// Set whether quantization is propagated through operations.
    pub fn with_propagation(mut self, propagation: QuantPropagation) -> Self {
        self.propagation = propagation;
        self
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
/// The precision of accumulating elements.
pub enum QuantAcc {
    /// Full precision.
    F32,
    /// Half precision.
    F16,
    /// bfloat16 precision.
    BF16,
}

/// Specify if the output of an operation is quantized using the scheme of the input
/// or returned unquantized.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantPropagation {
    /// The output is quantized using the scheme of the input.
    Propagate,
    /// The output is not quantized.
    Inhibit,
}

impl QuantSettings {
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
            Calibration::MinMax => match self.scheme.level {
                QuantLevel::Tensor => (B::float_min(tensor.clone()), B::float_max(tensor)),
                QuantLevel::Block(block_size) => {
                    let shape = tensor.shape();
                    let numel = shape.num_elements();

                    assert_eq!(
                        numel % block_size,
                        0,
                        "Tensor {shape:?} must be evenly divisible by block size {block_size}"
                    );

                    let num_blocks = numel / block_size;

                    let blocks = B::float_reshape(tensor, Shape::new([num_blocks, block_size]));
                    let blocks_min = B::float_reshape(
                        B::float_min_dim(blocks.clone(), 1),
                        Shape::new([num_blocks]),
                    );
                    let blocks_max =
                        B::float_reshape(B::float_max_dim(blocks, 1), Shape::new([num_blocks]));
                    (blocks_min, blocks_max)
                }
            },
        }
    }

    /// Compute the quantization parameters.
    pub fn compute_q_params<B: Backend>(
        &self,
        range: CalibrationRange<B>,
    ) -> QuantizationParameters<B> {
        match self.scheme {
            QuantScheme {
                level: QuantLevel::Tensor | QuantLevel::Block(_),
                mode: QuantMode::Symmetric,
                value: QuantValue::QInt8,
                ..
            } => {
                // Quantized range `[a, b]`
                let b = i8::MAX as i32;
                let a = -b;

                // Compute scale to convert an input value in range `[-alpha, alpha]`
                let values_range = range.min.abs().max_pair(range.max.abs()).mul_scalar(2);

                QuantizationParameters {
                    scales: values_range.div_scalar(b - a),
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
