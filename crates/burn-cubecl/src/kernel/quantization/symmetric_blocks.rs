use burn_backend::{
    TensorMetadata,
    cubecl::{dtype_to_elem_type, dtype_to_storage_type},
    quantization::QuantScheme,
};
use burn_std::{QuantMode, QuantStore, QuantValue};
use cubecl::{
    calculate_cube_count_elemwise,
    ir::types::Fp8Format,
    post_processing::minifloat::f32_to_fp8_bits,
    prelude::*,
    server::MemoryLayoutStrategy,
    std::{
        quant::{fp4::float_to_e2m1_bits, round::round_up_to_dtype},
        tensor::layout::linear::{LinearView, LinearViewMut},
    },
};

use crate::{
    kernel::{into_contiguous, utils::address_type},
    ops::empty_qtensor,
    tensor::CubeTensor,
};

/// One block quantized per unit: its largest magnitude times `inverse_range` is its scale,
/// rounded up to the scale's dtype and stored, and its values over the stored scale, clamped to
/// the format's range, packed `num_quants` to a word from the low bits.
///
/// A block is `block` consecutive elements of the contiguous input, which is what one spanning
/// only the innermost axis is; block `ABSOLUTE_POS`'s scale is element `ABSOLUTE_POS` of the
/// scales and its values the `block / num_quants` words from `ABSOLUTE_POS * block / num_quants`.
/// The input is read `num_quants` at a time, so each read packs into one word.
#[cube(launch_unchecked, address_type = "dynamic")]
fn quantize_symmetric_blocks_kernel<F: Float, FS: Numeric, N: Size>(
    input: LinearView<'_, Vector<F, N>>,
    mut values: LinearViewMut<'_, u32>,
    mut scales: LinearViewMut<'_, FS>,
    inverse_range: InputScalar,
    #[comptime] block: usize,
    #[comptime] scheme: QuantScheme,
    #[define(F, FS)] _dtypes: [ElemType; 2],
) {
    if !scales.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }
    let words = comptime!(block / scheme.num_quants());
    let first = ABSOLUTE_POS * words;
    let mut most = Vector::<F, N>::new(F::new(0.0));
    #[unroll]
    for at in 0..words {
        most = max(most, input.read(first + at).abs());
    }
    let mut largest = most.extract(0usize);
    #[unroll]
    for lane in 1..comptime!(scheme.num_quants()) {
        largest = max(largest, most.extract(lane));
    }

    // Rounded up rather than to nearest, as the composed path's quantize rounds it, so a block's
    // largest value never clips.
    let scale = FS::cast_from(round_up_to_dtype::<F>(
        largest * inverse_range.get::<F>(),
        scheme.scale_dtype(),
    ));
    scales.write(ABSOLUTE_POS, scale);
    let divisor = F::cast_from(scale);
    // A block of zeros under a zero scale is zeros; dividing would make it NaN.
    let inverse = select(divisor > F::new(0.0), F::new(1.0) / divisor, F::new(0.0));
    let (low, high) = comptime!(scheme.value.range());
    let (low, high) = comptime!((low as f64, high as f64));
    #[unroll]
    for at in 0..words {
        let scaled = input.read(first + at) * Vector::new(inverse);
        let snapped = clamp(
            round_to_grid::<F, N>(scaled, scheme.value),
            Vector::new(F::new(low)),
            Vector::new(F::new(high)),
        );
        values.write(first + at, pack::<F, N>(snapped, scheme.value));
    }
}

/// `value` rounded onto an integer format's grid; a minifloat rounds where it is encoded, against
/// the codes it has, so it passes through.
#[cube]
fn round_to_grid<F: Float, N: Size>(
    value: Vector<F, N>,
    #[comptime] quant: QuantValue,
) -> Vector<F, N> {
    match quant {
        QuantValue::E2M1 | QuantValue::E4M3 | QuantValue::E5M2 => value,
        _ => Vector::round(value),
    }
}

/// `value`'s codes, one field of the format's width each, packed into a word from the low bits.
#[cube]
fn pack<F: Float, N: Size>(value: Vector<F, N>, #[comptime] quant: QuantValue) -> u32 {
    let bits = comptime!(quant.size_bits() as u32);
    let mask = comptime!((1u32 << bits) - 1);
    let codes = match quant {
        QuantValue::E2M1 => float_to_e2m1_bits::<F, N>(value),
        QuantValue::E4M3 => f32_to_fp8_bits::<N>(Vector::cast_from(value), Fp8Format::E4M3),
        QuantValue::E5M2 => f32_to_fp8_bits::<N>(Vector::cast_from(value), Fp8Format::E5M2),
        _ => Vector::<u32, N>::reinterpret(Vector::<i32, N>::cast_from(value)),
    };
    let mut word = 0u32;
    #[unroll]
    for lane in 0..N::value() {
        word |= (codes.extract(lane) & mask) << comptime!(lane as u32 * bits);
    }
    word
}

/// `tensor` quantized under `scheme` in one launch, its block scales computed by min-max
/// calibration as the composed path computes them: each block's largest magnitude over the
/// format's.
///
/// `None` for a scheme it does not quantize alone — a per-tensor level, which normalizes the
/// blocks against the largest of them; a block spanning more than the innermost axis; a mode other
/// than symmetric; values not packed into words, or packed more to a word than the device reads in
/// one vector — which the composed calibration quantizes instead.
///
/// The output is laid out contiguously, values and scales both, so a reshape that merges or
/// splits its leading axes stays a change of metadata.
pub fn quantize_symmetric_blocks(tensor: &CubeTensor, scheme: &QuantScheme) -> Option<CubeTensor> {
    let block = scheme.block_size()?;
    let rank = tensor.meta.shape().num_dims();
    let dims = block.to_dim_vec(rank);
    let (inner, outer) = dims.split_last()?;
    let block = *inner as usize;
    let shape = tensor.shape();
    let num_quants = scheme.num_quants();
    let innermost = outer.iter().all(|extent| *extent == 1)
        && block > 0
        && block.is_multiple_of(num_quants)
        && shape[rank - 1].is_multiple_of(block);
    let packed = matches!(scheme.store, QuantStore::PackedU32(0));
    // A word's values are read as one vector, so the device has to read vectors that wide.
    let readable = tensor
        .client
        .io_optimized_vector_sizes(tensor.dtype.size())
        .any(|size| size == num_quants);
    if !innermost
        || !packed
        || !readable
        || scheme.tensor_scale().is_some()
        || !matches!(scheme.mode, QuantMode::Symmetric)
    {
        return None;
    }

    let input = into_contiguous(tensor.clone());
    let output = empty_qtensor(
        shape.clone(),
        *scheme,
        &input.device,
        MemoryLayoutStrategy::Contiguous,
    );
    let (values, scales) = output
        .quantized_handles()
        .expect("a block-quantized tensor");
    let blocks = shape.num_elements() / block;
    let (low, high) = scheme.value.range();
    let input_dtype = input.dtype;
    let dtype = dtype_to_storage_type(input_dtype);
    let inverse_range = InputScalar::new(2.0 / (high - low), dtype);
    let cube_dim = CubeDim::new(&input.client, blocks);
    let cube_count = calculate_cube_count_elemwise(&input.client, blocks, cube_dim);
    let at = address_type!(input, values, scales);
    let scale_dtype = ElemType::from_scale_dtype(scheme.scale_dtype());

    // SAFETY: each unit reads its block's `block` elements and writes its block's words, which
    // lie within the input and the values since the blocks tile both, and its one scale, bounded
    // by the scales' extent.
    unsafe {
        quantize_symmetric_blocks_kernel::launch_unchecked(
            &output.client,
            cube_count,
            cube_dim,
            at,
            num_quants,
            input.into_linear_view(),
            values.into_linear_view(),
            scales.into_linear_view(),
            inverse_range,
            block,
            *scheme,
            [dtype_to_elem_type(input_dtype), scale_dtype],
        );
    }

    Some(output)
}
