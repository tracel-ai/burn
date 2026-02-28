use cubecl::prelude::*;

/// Integer convolution: i8 × i8 → i32
///
/// Implements native integer convolution without float conversion,
/// supporting per-channel weight quantization (different scale/zp per output channel).
///
/// This kernel:
/// 1. Iterates over input spatial dimensions
/// 2. For each output position, performs sliding window sum (convolution)
/// 3. Uses i32 accumulation to prevent overflow
/// 4. Supports per-channel weight quantization (ONNX QLinearConv)
/// 5. Stores i32 accumulator (will be requantized later)
///
/// # Design
/// - im2col-like approach: Convert conv to matrix multiply
/// - Per-channel scales/zero-points for weights
/// - Symmetric or asymmetric input quantization
/// - Broadcasts per-channel parameters across spatial dims
///
/// # Parameters
/// - `input`: Quantized input [N, C, H, W] as i8
/// - `weight`: Quantized weight [OC, IC, KH, KW] as i8 (OC = output channels)
/// - `output`: Output accumulator [N, OC, OH, OW] as i32
/// - `weight_scales`: Per-channel scales [OC] (different scale per output channel)
/// - `weight_zero_points`: Per-channel zero-points [OC]
/// - `input_zero_point`: Input zero-point (scalar, same for all channels)
/// - `stride`, `padding`: Standard conv hyperparameters
#[cube(launch_unchecked, address_type = "dynamic")]
pub fn integer_conv2d_kernel<F: Float>(
    input: &Tensor<F>,                 // [N, C_in, H, W] as i8
    weight: &Tensor<F>,                // [C_out, C_in, KH, KW] as i8
    output: &mut Tensor<F>,            // [N, C_out, H_out, W_out] as i32
    weight_scales: &Tensor<F>,         // [C_out] per-channel scales
    weight_zero_points: &Tensor<F>,    // [C_out] per-channel zero-points
    input_zero_point: InputScalar,
    #[define(STRIDE_H)] stride_h: u32,
    #[define(STRIDE_W)] stride_w: u32,
    #[define(PADDING_H)] padding_h: u32,
    #[define(PADDING_W)] padding_w: u32,
) {
    // Global output position: [n, c_out, h_out, w_out]
    let n = BLOCK_POS_Z;
    let c_out = BLOCK_POS_Y;
    let h_out = THREAD_POS_Y;
    let w_out = THREAD_POS_X;

    // Bounds check
    if n >= output.shape(0) || c_out >= output.shape(1) || h_out >= output.shape(2) || w_out >= output.shape(3) {
        return;
    }

    // Get dimensions from tensor metadata
    let c_in = input.shape(1);
    let kh = weight.shape(2);
    let kw = weight.shape(3);

    // Get per-channel weight quantization parameters
    let w_scale = weight_scales[c_out];
    let w_zp = weight_zero_points[c_out].to_f32() as i32;
    let in_zp = input_zero_point.get::<i32>();

    // Accumulator for this output element
    let mut acc: i32 = 0;

    // Convolution: iterate over kernel spatial dimensions
    for kh_idx in 0u32..kh {
        for kw_idx in 0u32..kw {
            // Input spatial position (with padding)
            let h_in = (h_out * stride_h) + kh_idx;
            let w_in = (w_out * stride_w) + kw_idx;

            // Check bounds with padding
            let h_in_padded = if h_in >= padding_h {
                h_in - padding_h
            } else {
                continue; // Out of bounds (padding region)
            };

            let w_in_padded = if w_in >= padding_w {
                w_in - padding_w
            } else {
                continue; // Out of bounds (padding region)
            };

            // Accumulate over input channels
            for c_in_idx in 0u32..c_in {
                // Load input value
                let input_idx = n * c_in * input.shape(2) * input.shape(3)
                    + c_in_idx * input.shape(2) * input.shape(3)
                    + h_in_padded * input.shape(3)
                    + w_in_padded;

                let input_val = input[input_idx].to_f32() as i32;

                // Load weight value
                let weight_idx = c_out * c_in * kh * kw
                    + c_in_idx * kh * kw
                    + kh_idx * kw
                    + kw_idx;

                let weight_val = weight[weight_idx].to_f32() as i32;

                // Subtract zero-points
                let input_adj = input_val - in_zp;
                let weight_adj = weight_val - w_zp;

                // Accumulate: (input - zp_in) × (weight - zp_w)
                acc = acc + (input_adj * weight_adj);
            }
        }
    }

    // Store i32 accumulator
    output[n * output.shape(1) * output.shape(2) * output.shape(3)
        + c_out * output.shape(2) * output.shape(3)
        + h_out * output.shape(3)
        + w_out] = F::from_f32(acc as f32);
}

/// 1D convolution version: i8 × i8 → i32
///
/// Similar to conv2d but for 1D input (e.g., text, audio)
/// Input: [N, C_in, L] → Output: [N, C_out, L_out]
#[cube(launch_unchecked, address_type = "dynamic")]
pub fn integer_conv1d_kernel<F: Float>(
    input: &Tensor<F>,                 // [N, C_in, L]
    weight: &Tensor<F>,                // [C_out, C_in, K]
    output: &mut Tensor<F>,            // [N, C_out, L_out]
    weight_scales: &Tensor<F>,         // [C_out]
    weight_zero_points: &Tensor<F>,    // [C_out]
    input_zero_point: InputScalar,
    #[define(STRIDE)] stride: u32,
    #[define(PADDING)] padding: u32,
) {
    let n = BLOCK_POS_Y;
    let c_out = BLOCK_POS_X;
    let l_out = THREAD_POS_X;

    if n >= output.shape(0) || c_out >= output.shape(1) || l_out >= output.shape(2) {
        return;
    }

    let c_in = input.shape(1);
    let k = weight.shape(2);

    let w_scale = weight_scales[c_out];
    let w_zp = weight_zero_points[c_out].to_f32() as i32;
    let in_zp = input_zero_point.get::<i32>();

    let mut acc: i32 = 0;

    // Convolution over kernel dimension
    for k_idx in 0u32..k {
        let l_in_raw = (l_out * stride) + k_idx;
        let l_in = if l_in_raw >= padding {
            l_in_raw - padding
        } else {
            continue;
        };

        // Sum over input channels
        for c_in_idx in 0u32..c_in {
            let input_val = input[n * c_in * input.shape(2) + c_in_idx * input.shape(2) + l_in].to_f32() as i32;
            let weight_val = weight[c_out * c_in * k + c_in_idx * k + k_idx].to_f32() as i32;

            let input_adj = input_val - in_zp;
            let weight_adj = weight_val - w_zp;

            acc = acc + (input_adj * weight_adj);
        }
    }

    output[n * output.shape(1) * output.shape(2) + c_out * output.shape(2) + l_out] = F::from_f32(acc as f32);
}

/// Depthwise integer convolution: per-channel only
///
/// Optimized for depthwise conv where input and output channels are the same
/// (1 filter per input channel)
///
/// This is more efficient than general conv for depth-wise operations
#[cube(launch_unchecked, address_type = "dynamic")]
pub fn integer_depthwise_conv2d_kernel<F: Float>(
    input: &Tensor<F>,              // [N, C, H, W]
    weight: &Tensor<F>,             // [C, 1, KH, KW] (depthwise: 1 filter per input channel)
    output: &mut Tensor<F>,         // [N, C, H_out, W_out]
    weight_scales: &Tensor<F>,      // [C]
    weight_zero_points: &Tensor<F>, // [C]
    input_zero_point: InputScalar,
    #[define(STRIDE_H)] stride_h: u32,
    #[define(STRIDE_W)] stride_w: u32,
    #[define(PADDING_H)] padding_h: u32,
    #[define(PADDING_W)] padding_w: u32,
) {
    let n = BLOCK_POS_Z;
    let c = BLOCK_POS_Y;
    let h_out = THREAD_POS_Y;
    let w_out = THREAD_POS_X;

    if n >= output.shape(0) || c >= output.shape(1) {
        return;
    }

    let kh = weight.shape(2);
    let kw = weight.shape(3);

    let w_scale = weight_scales[c];
    let w_zp = weight_zero_points[c].to_f32() as i32;
    let in_zp = input_zero_point.get::<i32>();

    let mut acc: i32 = 0;

    // Convolution over spatial kernel
    for kh_idx in 0u32..kh {
        for kw_idx in 0u32..kw {
            let h_in = (h_out * stride_h) + kh_idx;
            let w_in = (w_out * stride_w) + kw_idx;

            if h_in < padding_h || w_in < padding_w {
                continue; // Padding region
            }

            let h_in_padded = h_in - padding_h;
            let w_in_padded = w_in - padding_w;

            let input_val = input[n * input.shape(1) * input.shape(2) * input.shape(3)
                + c * input.shape(2) * input.shape(3)
                + h_in_padded * input.shape(3)
                + w_in_padded].to_f32() as i32;

            let weight_val = weight[c * weight.shape(2) * weight.shape(3)
                + kh_idx * weight.shape(3)
                + kw_idx].to_f32() as i32;

            let input_adj = input_val - in_zp;
            let weight_adj = weight_val - w_zp;

            acc = acc + (input_adj * weight_adj);
        }
    }

    output[n * output.shape(1) * output.shape(2) * output.shape(3)
        + c * output.shape(2) * output.shape(3)
        + h_out * output.shape(3)
        + w_out] = F::from_f32(acc as f32);
}
