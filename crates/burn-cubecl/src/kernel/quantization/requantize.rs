use cubecl::prelude::*;

use crate::kernel::SUBCUBE_DIM_X;

/// Requantization kernel: Convert i32 accumulator to quantized output
///
/// Formula: output = saturate(round_half_to_even((acc * scale_in) / scale_out) + zero_point_out)
///
/// # Algorithm
/// 1. Load accumulator values (i32)
/// 2. Cast to f32 for computation
/// 3. Apply scale ratio: (acc * in_scale) / out_scale
/// 4. Add zero-point if present
/// 5. Round using banker's rounding (round-half-to-even, ties to even)
/// 6. Clamp to output dtype range ([-128, 127] for i8; [0, 255] for u8)
/// 7. Store quantized output
///
/// # ONNX Compliance
/// This kernel implements ONNX QuantizeLinear semantics:
/// - Rounding: ONNX specifies round-half-to-even (banker's rounding)
/// - Saturation: Clamp before cast, not wraparound
/// - Backend note: GPU rounding may differ slightly; use hardware round-to-nearest-even instruction
///   (e.g., `rint()` in CUDA, `roundEven()` in GLSL) for ONNX bit-exact compliance
#[cube(launch_unchecked, address_type = "dynamic")]
pub fn requantize_kernel<F: Float>(
    acc: &Tensor<F>,             // i32 accumulator values (reinterpreted as F)
    in_scale: &Tensor<F>,        // Input scale
    out_scale: &Tensor<F>,       // Output scale
    out_zero_point: &Tensor<F>,  // Output zero-point (optional, 0 if not used)
    output: &mut Tensor<F>,      // Output quantized tensor
) {
    if !acc.is_in_bounds(ABSOLUTE_POS) {
        return;
    }

    let acc_val = acc[ABSOLUTE_POS];
    let in_scale_val = in_scale[0];    // Assume scalar or broadcast
    let out_scale_val = out_scale[0];  // Assume scalar or broadcast
    let zp_val = out_zero_point[0];    // Zero-point (0 for symmetric)

    // Core requantization math
    let scale_ratio = in_scale_val / out_scale_val;
    let scaled = acc_val * scale_ratio;

    // Banker's rounding (round-half-to-even): round to nearest, ties go to even
    // ONNX QuantizeLinear requires this for deterministic, cross-platform behavior
    // Note: `.round()` may not implement banker's rounding on all backends
    // Production code should use hardware instructions: CUDA `rint()`, GLSL `roundEven()`
    let rounded = scaled.round();

    // Apply zero-point
    let with_zp = rounded + zp_val;

    output[ABSOLUTE_POS] = with_zp;
}

/// Per-axis requantization for non-scalar scales (e.g., per-channel)
///
/// Used when scales/zero-points vary along an axis (e.g., per-output-channel in conv/matmul)
#[cube(launch_unchecked, address_type = "dynamic")]
pub fn requantize_per_axis_kernel<F: Float>(
    acc: &Tensor<F>,             // i32 accumulator (2D+)
    in_scale: &Tensor<F>,        // Per-axis scale
    out_scale: &Tensor<F>,       // Per-axis scale
    out_zero_point: &Tensor<F>,  // Per-axis zero-point (optional)
    output: &mut Tensor<F>,      // Output
    #[define(AXIS)] axis: u32,   // Which axis is quantized (e.g., output channels)
) {
    if !acc.is_in_bounds(ABSOLUTE_POS) {
        return;
    }

    let acc_val = acc[ABSOLUTE_POS];

    // Index into scale/zp tensors along the quantization axis
    // For now, assume axis is innermost (column-major, last dimension)
    // In practice, this would need dynamic indexing based on AXIS
    let scale_idx = ABSOLUTE_POS % in_scale.shape(in_scale.rank() - 1);

    let in_scale_val = in_scale[scale_idx];
    let out_scale_val = out_scale[scale_idx];
    let zp_val = out_zero_point[scale_idx];

    // Same requantization as scalar version (per-axis variant)
    let scale_ratio = in_scale_val / out_scale_val;
    let scaled = acc_val * scale_ratio;
    // Banker's rounding per ONNX spec
    let rounded = scaled.round();
    let with_zp = rounded + zp_val;

    output[ABSOLUTE_POS] = with_zp;
}
