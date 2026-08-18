pub use burn_std::{BlockGrid, QPARAM_ALIGN, params_shape};
use burn_std::{FloatDType, QuantScheme, ScaleDtype, Shape, quantization::global_scale_dtype};

use super::{Calibration, QuantizationParametersPrimitive};
use crate::{Backend, TensorMetadata, get_device_settings};

/// One value per block of `scheme`'s grid: `tensor` viewed as `[g0, b0, g1, b1, ...]`, every
/// block axis folded by `reduce`, and the result in the grid's shape ([`params_shape`]).
///
/// A block is a rectangle, not a run of elements: `[2, 2]` on a `[2, 4]` tensor holds
/// `[a, b, e, f]`, so chunking the flat storage would only be right for a block that spans the
/// trailing dimension.
fn reduce_blocks<B: Backend>(
    tensor: B::FloatTensorPrimitive,
    scheme: &QuantScheme,
    reduce: impl Fn(B::FloatTensorPrimitive, usize) -> B::FloatTensorPrimitive,
) -> B::FloatTensorPrimitive {
    let shape = tensor.shape();
    let block = scheme
        .block_size()
        .expect("only a block scheme has blocks to reduce");
    let block = block.to_dim_vec(shape.num_dims());
    let mut split = Vec::with_capacity(2 * shape.num_dims());
    for (&dim, &extent) in shape.iter().zip(&block) {
        let extent = extent as usize;
        assert!(
            dim.is_multiple_of(extent),
            "Tensor {shape:?} must be evenly divisible by block size {block:?}"
        );
        split.push(dim / extent);
        split.push(extent);
    }
    let mut blocks = B::float_reshape(tensor, Shape::from(split.clone()));
    // Reductions keep their axis at size 1, so the block axes stay where they were.
    for axis in (1..split.len()).step_by(2) {
        blocks = reduce(blocks, axis);
    }
    B::float_reshape(blocks, params_shape(&shape, scheme))
}

/// Compute the quantization range mapping.
pub fn compute_range<B: Backend>(
    scheme: &QuantScheme,
    tensor: B::FloatTensorPrimitive,
    calibration: &Calibration,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
    match calibration {
        Calibration::MinMax => match scheme.block_size() {
            None => (B::float_min(tensor.clone()), B::float_max(tensor)),
            Some(_) => (
                reduce_blocks::<B>(tensor.clone(), scheme, B::float_min_dim),
                reduce_blocks::<B>(tensor, scheme, B::float_max_dim),
            ),
        },
        Calibration::AbsMean => {
            // gamma = mean(|W|) per tensor or block — symmetric range [-gamma, +gamma]
            assert!(
                global_scale_dtype(scheme).is_none(),
                "AbsMean calibration has no two-level form: BitNet's gamma is a mean over \
                 the whole tensor or block, which a per-tensor scale cannot decompose"
            );
            let gamma = match scheme.block_size() {
                None => B::float_mean(B::float_abs(tensor)),
                Some(_) => reduce_blocks::<B>(B::float_abs(tensor), scheme, B::float_mean_dim),
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

    let (scales, global) = match global_scale_dtype(scheme) {
        None => (scales, None),
        Some(_) => {
            let (scales, global) = normalize_scales::<B>(scales, scheme.scale_dtype());
            (scales, Some(global))
        }
    };

    QuantizationParametersPrimitive { scales, global }
}

/// Splits block scales into a per-tensor scale and block scales relative to it, returned as
/// `(scales, global)`. `global` is `f32`: at the block scales' precision it would land among the
/// subnormals, keeping a number of bits that depends on the weights' magnitude.
fn normalize_scales<B: Backend>(
    scales: B::FloatTensorPrimitive,
    block_dtype: ScaleDtype,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
    // Dividing by the block dtype's maximum is what lands the largest block scale exactly on it,
    // so a dtype whose maximum reaches f32's range drives the per-tensor scale subnormal instead.
    assert!(
        block_dtype.max_representable() <= burn_std::f16::MAX.to_f32(),
        "block scales in {block_dtype:?} reach f32's range, which a per-tensor scale cannot absorb"
    );

    let dtype = scales.dtype().into();
    let scales_f32 = B::float_cast(scales, FloatDType::F32);

    let global = B::float_div_scalar(
        B::float_max(scales_f32.clone()),
        block_dtype.max_representable().into(),
    );
    // Guards `0 / 0` for an all-zero tensor, and an underflow to zero for a very small one.
    let global = B::float_clamp_min(global, f32::MIN_POSITIVE.into());

    let broadcast = Shape::from(core::iter::repeat_n(1usize, scales_f32.shape().num_dims()));
    let scales = B::float_div(scales_f32, B::float_reshape(global.clone(), broadcast));

    (B::float_cast(scales, dtype), global)
}
