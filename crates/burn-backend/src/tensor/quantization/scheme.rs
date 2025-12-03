use burn_std::{QuantLevel, QuantMode, QuantScheme, Shape};

use super::Calibration;
use crate::{
    Backend, TensorMetadata, element::ElementConversion, tensor::QuantizationParametersPrimitive,
};

/// Calculate the shape of the quantization parameters for a given tensor and level
pub fn params_shape(data_shape: &Shape, level: QuantLevel) -> Shape {
    match level {
        QuantLevel::Tensor => Shape::new([1]),
        QuantLevel::Block(block_size) => {
            let mut params_shape = data_shape.clone();
            let block_size = block_size.to_dim_vec(data_shape.num_dims());

            for (shape, block_size) in params_shape.dims.iter_mut().zip(block_size) {
                *shape = (*shape).div_ceil(block_size as usize);
            }

            params_shape
        }
    }
}

/// Compute the quantization range mapping.
pub fn compute_range<B: Backend>(
    scheme: &QuantScheme,
    tensor: B::FloatTensorPrimitive,
    calibration: &Calibration,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
    match calibration {
        Calibration::MinMax => match scheme.level {
            QuantLevel::Tensor => (B::float_min(tensor.clone()), B::float_max(tensor)),
            QuantLevel::Block(block_size) => {
                let block_elems = block_size.num_elements();
                let shape = tensor.shape();
                let numel = shape.num_elements();

                assert_eq!(
                    numel % block_elems,
                    0,
                    "Tensor {shape:?} must be evenly divisible by block size {block_elems}"
                );

                let num_blocks = numel / block_elems;

                let params_shape = params_shape(&shape, scheme.level);

                let blocks = B::float_reshape(tensor, Shape::new([num_blocks, block_elems]));
                let blocks_min =
                    B::float_reshape(B::float_min_dim(blocks.clone(), 1), params_shape.clone());
                let blocks_max = B::float_reshape(B::float_max_dim(blocks, 1), params_shape);
                (blocks_min, blocks_max)
            }
        },
    }
}

/// Compute the quantization parameters.
pub fn compute_q_params<B: Backend>(
    scheme: &QuantScheme,
    min: B::FloatTensorPrimitive,
    max: B::FloatTensorPrimitive,
) -> QuantizationParametersPrimitive<B> {
    match scheme {
        QuantScheme {
            level: QuantLevel::Tensor | QuantLevel::Block(_),
            mode: QuantMode::Symmetric,
            ..
        } => {
            // Quantized range `[a, b]`
            let (a, b) = scheme.value.range();

            // Compute scale to convert an input value in range `[-alpha, alpha]`
            let min_abs = B::float_abs(min);
            let max_abs = B::float_abs(max);

            // `min_abs.max_pair(max_abs)`
            let mask = B::float_lower(min_abs.clone(), max_abs.clone());
            let values_range =
                B::float_mul_scalar(B::float_mask_where(min_abs, mask, max_abs), 2.elem());

            QuantizationParametersPrimitive {
                scales: B::float_div_scalar(values_range, (b - a).elem()),
            }
        }
    }
}
