use serde::{Deserialize, Serialize};

use crate::{Shape, Tensor, TensorMetadata, TensorPrimitive, backend::Backend};

use super::{
    Calibration, CalibrationRange, QuantizationParameters, QuantizationParametersPrimitive,
};

/// Describes a quantization scheme/configuration.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct QuantScheme {
    /// Granularity level of quantization (e.g., per-tensor).
    pub level: QuantLevel,
    /// Quantization mode (e.g., symmetric).
    pub mode: QuantMode,
    /// The logical data type of quantized input values (e.g., QInt8). This defines how values
    /// are interpreted during computation, independent of how they're stored (`q_store_type`).
    pub q_type: QuantInputType,
    /// Data type used for storing quantized values.
    pub q_store_type: QuantStoreType,
    /// Precision used for quantization parameters (e.g., scale).
    pub q_params_precision: QuantFloatPrecision,
    /// Precision used for accumulating intermediate values (e.g., during matmul).
    pub acc_precision: QuantFloatPrecision,
    /// Whether to propagate quantization to outputs or return unquantized results.
    pub propagation: QuantPropagation,
}

impl Default for QuantScheme {
    fn default() -> Self {
        Self {
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            q_store_type: QuantStoreType::U32,
            q_params_precision: QuantFloatPrecision::F32,
            acc_precision: QuantFloatPrecision::F32,
            propagation: QuantPropagation::Inhibit,
        }
    }
}

impl QuantScheme {
    /// Set the quantization level.
    pub fn set_level(mut self, level: QuantLevel) -> Self {
        self.level = level;
        self
    }

    /// Set the quantization mode.
    pub fn set_mode(mut self, mode: QuantMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the data type used for quantized values.
    pub fn set_q_type(mut self, q_type: QuantInputType) -> Self {
        self.q_type = q_type;
        self
    }

    /// Set the data type used to store quantized values.
    pub fn set_q_store_type(mut self, q_store_type: QuantStoreType) -> Self {
        self.q_store_type = q_store_type;
        self
    }

    /// Set the precision used for quantization parameters
    pub fn set_q_params_precision(mut self, q_params_precision: QuantFloatPrecision) -> Self {
        self.q_params_precision = q_params_precision;
        self
    }

    /// Set the accumulation precision used during computations.
    pub fn set_acc_precision(mut self, acc_precision: QuantFloatPrecision) -> Self {
        self.acc_precision = acc_precision;
        self
    }

    /// Set whether quantization is propagated through operations.
    pub fn set_propagation(mut self, propagation: QuantPropagation) -> Self {
        self.propagation = propagation;
        self
    }

    /// Returns the size of the quantization storage type in bits.
    pub fn size_bits_stored(&self) -> usize {
        match self.q_store_type {
            QuantStoreType::Native => self.q_type.size_bits(),
            QuantStoreType::U32 => 32,
            // QuantStoreType::U8 => 8,
        }
    }
}

/// Level or granularity of quantization.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantLevel {
    /// Quantize the whole tensor using a single tensor.
    Tensor,
    /// Quantize a tensor using multiple 1D linear blocks.
    Block(usize),
}

/// Data type used to represent quantized values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantInputType {
    /// 8-bit signed integer.
    QInt8,
}

impl QuantInputType {
    /// Returns the size of the quantization input type in bits.
    pub fn size_bits(&self) -> usize {
        match self {
            QuantInputType::QInt8 => 8,
        }
    }
}

/// Data type used to stored quantized values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantStoreType {
    /// Native quantization doesn't require packing and unpacking.
    Native,
    /// Store packed quantized values in a 4-byte unsigned integer.
    U32,
    // /// Store packed quantized values in a 8-bit unsigned integer.
    // U8,
}

/// Strategy used to quantize values.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantMode {
    /// Symmetric or scale quantization.
    Symmetric,
}

/// Quantization floating-point precision.
///
/// This is used to represent the floating-point precision of quantization parameters like the scale(s)
/// or the accumulation precision used during operations like matrix multiplication.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QuantFloatPrecision {
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

impl QuantScheme {
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
        match self {
            QuantScheme {
                level: QuantLevel::Tensor | QuantLevel::Block(_),
                mode: QuantMode::Symmetric,
                q_type: QuantInputType::QInt8,
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
