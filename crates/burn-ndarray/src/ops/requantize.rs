/// Pure Rust reference implementation of requantization
///
/// This module provides readable, verifiable requantization logic for testing
/// and as a reference for backend implementations.

/// Requantize an i32 accumulator to quantized output
///
/// Formula: output = saturate(round_half_to_even((acc * in_scale) / out_scale) + out_zp)
///
/// # Arguments
/// * `acc` - i32 accumulator value
/// * `in_scale` - Input scale factor (typically from matmul)
/// * `out_scale` - Output scale factor (target quantization)
/// * `out_zero_point` - Output zero-point (0 for symmetric)
/// * `out_min` - Minimum value for output dtype (e.g., -128 for i8)
/// * `out_max` - Maximum value for output dtype (e.g., 127 for i8)
///
/// # Returns
/// Requantized value as i32 (will be cast to actual dtype)
pub fn requantize_i32(
    acc: i32,
    in_scale: f32,
    out_scale: f32,
    out_zero_point: i32,
    out_min: i32,
    out_max: i32,
) -> i32 {
    // Step 1: Convert accumulator to float for computation
    let acc_float = acc as f64;

    // Step 2: Apply scale ratio
    // Using f64 for higher precision during computation
    let in_scale_f64 = in_scale as f64;
    let out_scale_f64 = out_scale as f64;
    let scale_ratio = in_scale_f64 / out_scale_f64;

    // Step 3: Scale the accumulator
    let scaled = acc_float * scale_ratio;

    // Step 4: Banker's rounding (round-half-to-even)
    // This ensures deterministic results across platforms
    let rounded = banker_round(scaled);

    // Step 5: Add zero-point
    let with_zp = rounded + out_zero_point as f64;

    // Step 6: Saturating cast to output range
    let result = if with_zp < out_min as f64 {
        out_min
    } else if with_zp > out_max as f64 {
        out_max
    } else {
        with_zp as i32
    };

    result
}

/// Banker's rounding (round-half-to-even)
///
/// Rounds to nearest integer. When exactly halfway between two integers,
/// rounds to the nearest even integer.
///
/// Examples:
/// - 2.3 → 2
/// - 2.5 → 2 (even)
/// - 3.5 → 4 (even)
/// - 2.7 → 3
fn banker_round(x: f64) -> f64 {
    let floor_x = x.floor();
    let frac = x - floor_x;

    if frac < 0.5 {
        floor_x
    } else if frac > 0.5 {
        floor_x + 1.0
    } else {
        // Exactly 0.5: round to even
        let int_part = floor_x as i64;
        if int_part % 2 == 0 {
            floor_x
        } else {
            floor_x + 1.0
        }
    }
}

/// Requantize per-axis (per-channel) quantization
///
/// Used when scales/zero-points vary along an axis (e.g., per-output-channel)
pub fn requantize_per_axis(
    acc: &[i32],
    in_scales: &[f32],
    out_scales: &[f32],
    out_zero_points: &[i32],
    out_min: i32,
    out_max: i32,
) -> Vec<i32> {
    acc.iter()
        .enumerate()
        .map(|(i, &acc_val)| {
            let scale_idx = i % in_scales.len();
            requantize_i32(
                acc_val,
                in_scales[scale_idx],
                out_scales[scale_idx],
                out_zero_points[scale_idx],
                out_min,
                out_max,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_requantize_basic() {
        // Accumulator: 1000 (from i8*i8 matmul)
        // Scales: 0.01 / 0.1 = 0.1
        // Result: 1000 * 0.1 + 0 = 100, clamped to [-128, 127] = 127
        let result = requantize_i32(1000, 0.01, 0.1, 0, -128, 127);
        assert_eq!(result, 127); // Saturated
    }

    #[test]
    fn test_requantize_with_zp() {
        // Test with asymmetric zero-point
        let result = requantize_i32(100, 0.1, 0.1, 10, -128, 127);
        assert_eq!(result, 110); // 100 * 1.0 + 10
    }

    #[test]
    fn test_banker_rounding_to_even() {
        // 2.5 should round to 2 (even)
        assert_eq!(banker_round(2.5), 2.0);

        // 3.5 should round to 4 (even)
        assert_eq!(banker_round(3.5), 4.0);

        // 2.3 should round to 2
        assert_eq!(banker_round(2.3), 2.0);

        // 2.7 should round to 3
        assert_eq!(banker_round(2.7), 3.0);

        // Negative: -2.5 should round to -2 (even)
        assert_eq!(banker_round(-2.5), -2.0);
    }

    #[test]
    fn test_requantize_saturation() {
        // Test lower bound saturation
        let result = requantize_i32(-1000, 0.01, 0.1, 0, -128, 127);
        assert_eq!(result, -128); // Saturated at minimum

        // Test upper bound saturation
        let result = requantize_i32(2000, 0.01, 0.1, 0, -128, 127);
        assert_eq!(result, 127); // Saturated at maximum
    }

    #[test]
    fn test_requantize_precision() {
        // Test that scale ratio is computed with high precision
        // 100000 * 0.001 / 1.0 = 100
        let result = requantize_i32(100000, 0.001, 1.0, 0, -32768, 32767);
        assert_eq!(result, 100);
    }

    #[test]
    fn test_requantize_per_axis() {
        let acc = vec![100, 200, 300];
        let in_scales = vec![0.1, 0.2, 0.3];
        let out_scales = vec![1.0, 2.0, 3.0];
        let out_zps = vec![0, 0, 0];

        let result = requantize_per_axis(&acc, &in_scales, &out_scales, &out_zps, -128, 127);

        assert_eq!(result[0], 10);  // 100 * 0.1 / 1.0 = 10
        assert_eq!(result[1], 20);  // 200 * 0.2 / 2.0 = 20
        assert_eq!(result[2], 30);  // 300 * 0.3 / 3.0 = 30
    }
}
