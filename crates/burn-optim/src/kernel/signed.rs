use cubecl::prelude::*;

/// Encode one normalized scalar in `[-1, 1]` to its signed dynamic 8-bit code.
///
/// Output layout (low to high bits):
///   - bits 0..N: fraction (N depends on depth bucket)
///   - bit  N:    indicator (the lowest set bit identifies the depth)
///   - bit  7:    sign (set when input is negative)
///
/// Inverse of `decode_signed_one` up to quantization error.
#[cube]
pub fn encode(normalized: f32) -> u32 {
    let abs_val = normalized.abs();

    // Treat anything below this threshold as exact zero.
    let is_zero = abs_val < 1e-7f32;
    if is_zero {
        u32::new(0)
    } else {
        // Sign bit: contributes 128 when negative.
        let is_negative = normalized < 0.0f32;
        let sign_contrib = if is_negative {
            u32::new(128)
        } else {
            u32::new(0)
        };

        // Determine depth bucket. Mirrors the lower_bits >= {64, 32, 16, ...} ladder
        // in decode_signed_one.
        let (max_frac_f, upper, lower, indicator_val) = if abs_val >= 0.1f32 {
            (63.0f32, 1.0f32, 0.1f32, u32::new(64))
        } else if abs_val >= 0.01f32 {
            (31.0f32, 0.1f32, 0.01f32, u32::new(32))
        } else if abs_val >= 0.001f32 {
            (15.0f32, 0.01f32, 0.001f32, u32::new(16))
        } else if abs_val >= 0.0001f32 {
            (7.0f32, 0.001f32, 0.0001f32, u32::new(8))
        } else if abs_val >= 0.00001f32 {
            (3.0f32, 0.0001f32, 0.00001f32, u32::new(4))
        } else if abs_val >= 0.000001f32 {
            (1.0f32, 0.00001f32, 0.000001f32, u32::new(2))
        } else {
            // Depth 6: range collapses, fraction is always 0.
            (0.0f32, 0.000001f32, 0.0000001f32, u32::new(1))
        };

        // Linear position within the bucket: t = (|x| - lower) / (upper - lower),
        // clamped to [0, 1]. The depth-6 bucket has max_frac_f = 0 so the fraction
        // collapses to 0 regardless; we still guard the divisor for safety.
        let range = upper - lower;
        let safe_range = if range > 1e-10f32 {
            range
        } else {
            1.0f32.into()
        };
        let t_raw = (abs_val - lower) / safe_range;
        let t = clamp(t_raw, 0.0f32, 1.0f32);

        // Quantize to integer fraction. round() then clamp belt-and-suspenders
        // matches the tensor-ops version exactly.
        let fraction_f = clamp((t * max_frac_f).round(), 0.0f32, 63.0f32);
        let fraction_raw = u32::cast_from(fraction_f);

        // Clamp fraction to the bucket's max_frac. Avoids the round() ever
        // pushing us into a higher bucket's bits.
        let max_frac_u = u32::cast_from(max_frac_f);
        let fraction = if fraction_raw > max_frac_u {
            max_frac_u
        } else {
            fraction_raw
        };

        sign_contrib + indicator_val + fraction
    }
}

/// Decode one signed dynamic 8-bit code to its normalized f32 value.
///
/// Inverse of [`encode_signed_one`]. Round-trip is exact for code 0 and
/// stable to within the per-bucket quantization step otherwise.
#[cube]
pub fn decode(code: u32) -> f32 {
    // Lower 7 bits hold indicator + fraction. Top bit is the sign.
    let lower_bits = code & 127u32;
    let is_zero = lower_bits == 0u32;

    if is_zero {
        f32::new(0.0)
    } else {
        // Sign from the top bit: 1 -> negative, 0 -> positive.
        let is_negative = (code & 128u32) != 0u32;
        let sign = if is_negative {
            f32::new(-1.0)
        } else {
            f32::new(1.0)
        };

        // Determine depth bucket from the position of the indicator bit.
        // Mirrors the encoder's bucket ladder exactly — same thresholds,
        // same upper/lower/max_frac/indicator values per bucket.
        let (max_frac_f, upper, lower, indicator_val) = if lower_bits >= 64u32 {
            (63.0f32, 1.0f32, 0.1f32, u32::new(64))
        } else if lower_bits >= 32u32 {
            (31.0f32, 0.1f32, 0.01f32, u32::new(32))
        } else if lower_bits >= 16u32 {
            (15.0f32, 0.01f32, 0.001f32, u32::new(16))
        } else if lower_bits >= 8u32 {
            (7.0f32, 0.001f32, 0.0001f32, u32::new(8))
        } else if lower_bits >= 4u32 {
            (3.0f32, 0.0001f32, 0.00001f32, u32::new(4))
        } else if lower_bits >= 2u32 {
            (1.0f32, 0.00001f32, 0.000001f32, u32::new(2))
        } else {
            (0.0f32, 0.000001f32, 0.0000001f32, u32::new(1))
        };

        // Isolate the fraction bits.
        let fraction = lower_bits - indicator_val;
        let fraction_f = f32::cast_from(fraction);

        // t = fraction / max_frac, in [0, 1]. Guard against the depth-6
        // bucket where max_frac is 0 — t collapses to 0 anyway because
        // fraction is also 0 (the bucket only has one valid code).
        let safe_denom = if max_frac_f > 0.0f32 {
            max_frac_f
        } else {
            1.0f32.into()
        };
        let t = fraction_f / safe_denom;

        // Linearly interpolate within the bucket: lower + t * (upper - lower).
        let range = upper - lower;
        let abs_val = lower + t * range;

        sign * abs_val
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::bytes::Bytes;
    use cubecl::prelude::*;
    use cubecl::wgpu::WgpuRuntime;

    /// The runtime used for kernel tests in this module.
    /// Swap this one line to retarget all tests at a different backend
    /// (e.g. `cubecl::cuda::CudaRuntime` for CUDA, or any other runtime).
    type TestRuntime = WgpuRuntime;

    #[cube(launch_unchecked)]
    fn roundtrip_kernel(
        input: &Array<f32>,
        encoded_out: &mut Array<u32>,
        decoded_out: &mut Array<f32>,
    ) {
        if ABSOLUTE_POS < input.len() {
            let code = encode(input[ABSOLUTE_POS]);
            encoded_out[ABSOLUTE_POS] = code;
            decoded_out[ABSOLUTE_POS] = decode(code);
        }
    }

    fn signed_roundtrip_via_kernel<R: Runtime>(
        client: &ComputeClient<R>,
        values: &[f32],
    ) -> (Vec<u32>, Vec<f32>) {
        let n = values.len();

        let input_bytes = f32::as_bytes(values).to_vec();
        let input_handle = client.create(Bytes::from_bytes_vec(input_bytes));

        let encoded_handle = client.empty(n * core::mem::size_of::<u32>());
        let decoded_handle = client.empty(n * core::mem::size_of::<f32>());

        let threads_per_cube: u32 = 64;
        let cubes = (n as u32).div_ceil(threads_per_cube);

        unsafe {
            roundtrip_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new(client, 1),
                ArrayArg::from_raw_parts(input_handle, n),
                ArrayArg::from_raw_parts(encoded_handle.clone(), n),
                ArrayArg::from_raw_parts(decoded_handle.clone(), n),
            );
        }

        let encoded_bytes: Bytes = client.read_one_unchecked(encoded_handle);
        let decoded_bytes: Bytes = client.read_one_unchecked(decoded_handle);

        let codes: Vec<u32> = u32::from_bytes(&encoded_bytes).to_vec();
        let decoded: Vec<f32> = f32::from_bytes(&decoded_bytes).to_vec();

        (codes, decoded)
    }

    #[test]
    fn test_signed_dynamic_quantization_roundtrip() {
        let client = TestRuntime::client(&Default::default());

        let values: Vec<f32> = vec![
            0.0, 0.5, -0.5, 1.0, -1.0, 0.1, -0.1, 0.01, -0.01, 0.001, 0.0001, 0.00001, 0.95, 0.05,
            0.005, 0.123, -0.456, 0.789,
        ];

        let (codes, decoded) = signed_roundtrip_via_kernel::<TestRuntime>(&client, &values);

        println!();
        for i in 0..values.len() {
            let orig = values[i];
            let reconstructed = decoded[i];
            let code = codes[i];
            let err = if orig.abs() > 1e-7 {
                ((reconstructed - orig) / orig).abs() * 100.0
            } else {
                (reconstructed - orig).abs() * 100.0
            };
            println!(
                "  {:.7} -> {:.7}  (code: {:3}, err: {:.2}%)",
                orig, reconstructed, code, err
            );
        }
    }
}
