// We re-export those types.
pub use cubecl_quant::scheme::{
    QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};

use serde::{Deserialize, Serialize};

use crate::{Shape, Tensor, TensorMetadata, TensorPrimitive, backend::Backend};

use super::{
    Calibration, CalibrationRange, QuantizationParameters, QuantizationParametersPrimitive,
};

#[derive(
    Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
/// The precision of accumulating elements.
pub enum QuantAcc {
    /// Full precision.
    #[default]
    F32,
    /// Half precision.
    F16,
    /// bfloat16 precision.
    BF16,
}

/// Specify if the output of an operation is quantized using the scheme of the input
/// or returned unquantized.
#[derive(
    Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default,
)]
pub enum QuantPropagation {
    /// The output is quantized using the scheme of the input.
    Propagate,
    /// The output is not quantized.
    #[default]
    Inhibit,
}

/// Compute the quantization range mapping.
pub fn compute_range<B: Backend, const D: usize>(
    scheme: &QuantScheme,
    tensor: &Tensor<B, D>,
    calibration: &Calibration,
) -> CalibrationRange<B> {
    let (min, max) = match &tensor.primitive {
        TensorPrimitive::Float(tensor) => {
            compute_range_primitive::<B>(scheme, tensor.clone(), calibration)
        }
        TensorPrimitive::QFloat(_) => unreachable!(),
    };

    CalibrationRange {
        min: Tensor::from_primitive(TensorPrimitive::Float(min)),
        max: Tensor::from_primitive(TensorPrimitive::Float(max)),
    }
}

pub(crate) fn compute_range_primitive<B: Backend>(
    scheme: &QuantScheme,
    tensor: B::FloatTensorPrimitive,
    calibration: &Calibration,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
    match calibration {
        Calibration::MinMax => match scheme.level {
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
    scheme: &QuantScheme,
    range: CalibrationRange<B>,
) -> QuantizationParameters<B> {
    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            ..
        } => {
            // Quantized range `[a, b]`
            let (a, b) = scheme.value.range();

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
    scheme: &QuantScheme,
    min: B::FloatTensorPrimitive,
    max: B::FloatTensorPrimitive,
) -> QuantizationParametersPrimitive<B> {
    let range = CalibrationRange {
        min: Tensor::from_primitive(TensorPrimitive::Float(min)),
        max: Tensor::from_primitive(TensorPrimitive::Float(max)),
    };
    compute_q_params(scheme, range).into()
}
