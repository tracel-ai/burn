use alloc::vec::Vec;
use burn_std::quantization::inner_level;
pub use burn_std::{BlockLayout, QPARAM_ALIGN, params_shape};
use burn_std::{
    FloatDType, QuantScheme, Shape,
    quantization::{BlockExtent, BlockScale, BlockSize},
};

use super::{Calibration, QuantizationParametersPrimitive};
use crate::{Backend, TensorMetadata, get_or_init_device_settings};

/// One value per block of `block` cells along each axis: `tensor` viewed as
/// `[n0, b0, n1, b1, ...]`, every block axis reduced by `reduce`, and the result shaped as the
/// grid of blocks, `[n0, n1, ...]`.
///
/// A block is a rectangle, not a run of elements: `[2, 2]` on a `[2, 4]` tensor holds
/// `[a, b, e, f]`, so chunking the flat storage would only be right for a block that spans the
/// trailing dimension.
fn reduce_blocks<B: Backend>(
    tensor: B::FloatTensorPrimitive,
    block: &[usize],
    reduce: impl Fn(B::FloatTensorPrimitive, usize) -> B::FloatTensorPrimitive,
) -> B::FloatTensorPrimitive {
    let shape = tensor.shape();
    let split = split_blocks(&shape, block);
    let mut blocks = B::float_reshape(tensor, split.clone());
    // Reductions keep their axis at size 1, so the block axes stay where they were; an axis
    // already at 1 has nothing to fold and would only cost a launch.
    for axis in (1..split.num_dims())
        .step_by(2)
        .filter(|&axis| split[axis] > 1)
    {
        blocks = reduce(blocks, axis);
    }
    let grid: Vec<usize> = split.iter().step_by(2).copied().collect();
    B::float_reshape(blocks, Shape::from(grid))
}

/// `shape` viewed as `[n0, b0, n1, b1, ...]`: `n` blocks of `b` cells along each axis.
fn split_blocks(shape: &Shape, block: &[usize]) -> Shape {
    let mut split = Vec::with_capacity(2 * shape.num_dims());
    for (&dim, &extent) in shape.iter().zip(block) {
        assert!(
            dim.is_multiple_of(extent),
            "Tensor {shape:?} must be evenly divisible by block size {block:?}"
        );
        split.push(dim / extent);
        split.push(extent);
    }
    Shape::from(split)
}

/// Compute the quantization range mapping.
pub fn compute_range<B: Backend>(
    scheme: &QuantScheme,
    tensor: B::FloatTensorPrimitive,
    calibration: &Calibration,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
    match calibration {
        Calibration::MinMax => match inner_level(scheme).size {
            BlockSize::TENSOR => (B::float_min(tensor.clone()), B::float_max(tensor)),
            size => {
                let block = size.values(tensor.shape().as_slice());
                (
                    reduce_blocks::<B>(tensor.clone(), &block, B::float_min_dim),
                    reduce_blocks::<B>(tensor, &block, B::float_max_dim),
                )
            }
        },
        Calibration::AbsMean => {
            // gamma = mean(|W|) per tensor or block — symmetric range [-gamma, +gamma]
            assert!(
                scheme.levels().nth(1).is_none(),
                "AbsMean calibration has no two-level form: BitNet's gamma is a mean over \
                 the whole tensor or block, which a coarser scale cannot decompose"
            );
            let gamma = match inner_level(scheme).size {
                BlockSize::TENSOR => B::float_mean(B::float_abs(tensor)),
                size => {
                    let block = size.values(tensor.shape().as_slice());
                    reduce_blocks::<B>(B::float_abs(tensor), &block, B::float_mean_dim)
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
    let bool_dtype = get_or_init_device_settings::<B>(&min.device()).bool_dtype;
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

    let (scales, coarser) = match scheme.levels().nth(1) {
        None => (scales, None),
        Some(coarser) => {
            let (scales, coarser) = normalize_scales::<B>(scales, &inner_level(scheme), &coarser);
            (scales, Some(coarser))
        }
    };

    QuantizationParametersPrimitive { scales, coarser }
}

/// Splits the finest scales into the coarser level's scales and finest scales relative to them,
/// returned as `(scales, coarser)`. Each coarser scale is the largest finest scale of its block
/// over where the finest dtype puts it
/// ([`largest_under_coarser`](burn_std::ScaleDtype::largest_under_coarser)). `coarser` is `f32`:
/// at the block scales' precision it would land among the subnormals, keeping a number of bits
/// that depends on the weights' magnitude.
///
/// `scales` is the finest grid, a rank-n grid of blocks or one element for a whole-tensor level;
/// `coarser` comes out as its own level's grid, one element for one scale for the whole tensor.
fn normalize_scales<B: Backend>(
    scales: B::FloatTensorPrimitive,
    finest: &BlockScale,
    coarser: &BlockScale,
) -> (B::FloatTensorPrimitive, B::FloatTensorPrimitive) {
    let dtype = scales.dtype().into();
    let scales_f32 = B::float_cast(scales, FloatDType::F32);
    let grid = scales_f32.shape();
    let block = coarser_block_in_grid(finest, coarser, &grid);

    let largest = reduce_blocks::<B>(scales_f32.clone(), &block, B::float_max_dim);
    let landing = finest.dtype.largest_under_coarser();
    // Guards `0 / 0` for an all-zero block, and an underflow to zero for a very small one.
    let coarser_scales = B::float_clamp_min(
        B::float_div_scalar(largest, landing.into()),
        f32::MIN_POSITIVE.into(),
    );

    // Each finest scale over its own block's coarser scale: both viewed as blocks of the coarser
    // level, the coarser one broadcast over the cells of its block.
    let split = split_blocks(&grid, &block);
    let mut one_cell = split.clone();
    for axis in (1..one_cell.num_dims()).step_by(2) {
        one_cell[axis] = 1;
    }
    let relative = B::float_div(
        B::float_reshape(scales_f32, split),
        B::float_reshape(coarser_scales.clone(), one_cell),
    );
    let scales = B::float_reshape(relative, grid);

    let coarser_scales = match coarser.size {
        BlockSize::TENSOR => B::float_reshape(coarser_scales, Shape::new([1])),
        _ => coarser_scales,
    };

    (B::float_cast(scales, dtype), coarser_scales)
}

/// How many cells of the finest level's `grid` a block of the coarser level covers along each
/// axis: the whole axis, or its values over the finest block's.
fn coarser_block_in_grid(finest: &BlockScale, coarser: &BlockScale, grid: &Shape) -> Vec<usize> {
    let rank = grid.num_dims();
    finest
        .size
        .extents_in_rank(rank)
        .zip(coarser.size.extents_in_rank(rank))
        .zip(grid.iter())
        .map(|((finest, coarser), &cells)| match (finest, coarser) {
            (_, BlockExtent::WholeAxis) => cells,
            (BlockExtent::Values(finest), BlockExtent::Values(coarser)) => {
                (coarser.get() / finest) as usize
            }
            // A coarser block covers whole finest blocks, so a whole-axis finest block has a
            // whole-axis coarser one over it.
            (BlockExtent::WholeAxis, BlockExtent::Values(_)) => cells,
        })
        .collect()
}
