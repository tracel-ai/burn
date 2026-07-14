use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Shape, SliceArg, quantization::QuantScheme, s};

/// Slice a quantized tensor (exercising `q_slice`) and compare against slicing the same
/// quantized data in full precision (dequantized) - within a permissive tolerance.
fn should_quantize_slice_dequantize_arange<const D: usize, S>(shape: [usize; D], slices: S)
where
    S: SliceArg + Clone,
{
    let numel = Shape::from(shape).num_elements() as i64;

    let device = Default::default();

    // Always use the default scheme when testing `q_slice`.
    let scheme = QuantScheme::default();

    let quantized = TestTensorInt::arange(0..numel, &device)
        .float()
        .div_scalar(numel)
        .reshape::<D, _>(shape)
        .quantize_dynamic(&scheme);

    // Reference: dequantize, then slice in full precision.
    let output_ref = quantized
        .clone()
        .dequantize()
        .slice(slices.clone())
        .into_data();

    // Slice the quantized tensor (exercises `q_slice`), then dequantize.
    let output = quantized.slice(slices).dequantize().into_data();

    output.assert_approx_eq::<FloatElem>(&output_ref, Tolerance::permissive());
}

// Note: the default `QuantScheme` stores `Q8F` values packed into `u32` (4 values per element),
// so the last dimension of both the input shape and the sliced output must be a multiple of 4.

#[test]
fn should_slice_1d_full() {
    should_quantize_slice_dequantize_arange([8], s![..]);
}

#[test]
fn should_slice_1d_range() {
    should_quantize_slice_dequantize_arange([8], s![2..6]);
}

#[test]
fn should_slice_2d_rows() {
    should_quantize_slice_dequantize_arange([4, 8], s![1..3, ..]);
}

#[test]
fn should_slice_2d_cols() {
    should_quantize_slice_dequantize_arange([4, 8], s![.., 0..4]);
}

#[test]
fn should_slice_2d_both_dims() {
    should_quantize_slice_dequantize_arange([4, 8], s![1..3, 4..8]);
}

#[test]
fn should_slice_3d() {
    should_quantize_slice_dequantize_arange([2, 3, 8], s![0..1, 1..3, ..]);
}

#[test]
fn should_slice_single_index_row() {
    // A single index keeps the rank, producing a dimension of size 1.
    should_quantize_slice_dequantize_arange([4, 8], s![1, ..]);
}

#[test]
fn should_slice_inclusive_range() {
    should_quantize_slice_dequantize_arange([4, 8], s![1..=2, ..]);
}

#[test]
fn should_slice_negative_index() {
    // Negative start counts from the end: last two rows.
    should_quantize_slice_dequantize_arange([4, 8], s![-2.., ..]);
}

#[test]
fn should_slice_negative_range_cols() {
    // Last 4 columns via a negative range.
    should_quantize_slice_dequantize_arange([4, 8], s![.., -4..]);
}

#[test]
fn should_slice_with_step_rows() {
    // Every other row.
    should_quantize_slice_dequantize_arange([8, 8], s![0..8;2, ..]);
}

#[test]
fn should_slice_with_step_cols() {
    // Every other column: output last dim is ceil(8 / 2) = 4.
    should_quantize_slice_dequantize_arange([4, 8], s![.., 0..8;2]);
}

#[test]
fn should_slice_reversed_rows() {
    // Reverse the first dimension (negative step).
    should_quantize_slice_dequantize_arange([4, 8], s![..;-1, ..]);
}

#[test]
fn should_slice_reversed_cols() {
    // Reverse the last dimension; size stays a multiple of 4.
    should_quantize_slice_dequantize_arange([4, 8], s![.., ..;-1]);
}

#[test]
fn should_slice_partial_dims() {
    // Fewer slices than dims: trailing dimension is sliced fully.
    should_quantize_slice_dequantize_arange([4, 8], s![1..3]);
}
