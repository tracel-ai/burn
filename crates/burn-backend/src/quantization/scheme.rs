use burn_std::{FloatDType, QuantLevel, QuantMode, QuantParam, QuantScheme, Shape};
pub use burn_std::{QPARAM_ALIGN, params_shape};

use super::{Calibration, QuantizationParametersPrimitive};
use crate::{Backend, TensorMetadata, get_device_settings};

/// Compute the quantization range mapping.
pub fn compute_range<B: Backend>(
    scheme: &QuantScheme,
    tensor: B::FloatTensorPrimitive,
    calibration: &Calibration,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
    match calibration {
        Calibration::MinMax => match scheme.level.block_size() {
            None => (B::float_min(tensor.clone()), B::float_max(tensor)),
            Some(block_size) => {
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
        Calibration::AbsMean => {
            // gamma = mean(|W|) per tensor or block — symmetric range [-gamma, +gamma]
            let gamma = match scheme.level {
                QuantLevel::BlockTensor { .. } => panic!(
                    "AbsMean calibration has no two-level form: BitNet's gamma is a mean over \
                     the whole tensor or block, which a per-tensor scale cannot decompose"
                ),
                QuantLevel::Tensor => B::float_mean(B::float_abs(tensor)),
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
                    let blocks = B::float_reshape(
                        B::float_abs(tensor),
                        Shape::new([num_blocks, block_elems]),
                    );
                    B::float_reshape(B::float_mean_dim(blocks, 1), params_shape)
                }
            };
            let neg_gamma = B::float_neg(gamma.clone());
            (neg_gamma, gamma)
        }
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
            mode: QuantMode::Symmetric,
            ..
        } => {
            let bool_dtype = get_device_settings::<B>(&min.device()).bool_dtype;
            // Quantized range `[a, b]`
            let (a, b) = scheme.value.range();

            // Compute scale to convert an input value in range `[-alpha, alpha]`
            let min_abs = B::float_abs(min);
            let max_abs = B::float_abs(max);

            // `min_abs.max_pair(max_abs)`
            let mask = B::float_lower(min_abs.clone(), max_abs.clone(), bool_dtype);
            let values_range =
                B::float_mul_scalar(B::float_mask_where(min_abs, mask, max_abs), 2f32.into());

            let scales = B::float_div_scalar(values_range, (b - a).into());

            let (scales, global) = match scheme.level.global_param() {
                None => (scales, None),
                Some(_) => {
                    let (scales, global) = normalize_scales::<B>(scales, scheme.param);
                    (scales, Some(global))
                }
            };
            QuantizationParametersPrimitive { scales, global }
        }
    }
}

/// Split block scales into a per-tensor scale and block scales relative to it.
///
/// The global is picked so the largest block scale lands at the top of `block_param`'s range, which
/// is what lets a narrow type cover a tensor whose raw scales would otherwise overflow or underflow
/// it. Both levels come back unrounded, as the one-level scales do; backends round each to the
/// precision it is stored in and quantize against that product.
///
/// The per-tensor scale is returned in `f32` whatever the tensor's dtype is. It sits a whole block
/// param's range below the block scales, so at `f16` it would land among the subnormals and keep a
/// number of bits that depends on the weights' magnitude, which is what the second level exists to
/// remove.
fn normalize_scales<B: Backend>(
    scales: B::FloatTensorPrimitive,
    block_param: QuantParam,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
    let dtype = scales.dtype().into();
    let scales_f32 = B::float_cast(scales, FloatDType::F32);

    let global = B::float_div_scalar(
        B::float_max(scales_f32.clone()),
        block_param.max_representable().into(),
    );
    // Guards `0 / 0` for an all-zero tensor, and an underflow to zero for a very small one.
    let global = B::float_clamp_min(global, f32::MIN_POSITIVE.into());

    let broadcast = Shape::from(vec![1usize; scales_f32.shape().num_dims()]);
    let scales = B::float_div(scales_f32, B::float_reshape(global.clone(), broadcast));

    (B::float_cast(scales, dtype), global)
}
