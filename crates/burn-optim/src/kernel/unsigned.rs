use cubecl::prelude::*;

#[cube]
pub fn encode(normalized: f32) -> u32 {
    // No abs() — matches tensor-ops behavior. Negative inputs satisfy
    // `normalized < 1e-7` and get encoded as 0, matching the unsigned
    // codec's domain assumption that inputs are non-negative.
    let is_zero = normalized < 1e-7f32;

    if is_zero {
        u32::new(0)
    } else {
        let (max_frac_f, upper, lower, indicator_val) = if normalized >= 0.1f32 {
            (127.0f32, 1.0f32, 0.1f32, u32::new(128))
        } else if normalized >= 0.01f32 {
            (63.0f32, 0.1f32, 0.01f32, u32::new(64))
        } else if normalized >= 0.001f32 {
            (31.0f32, 0.01f32, 0.001f32, u32::new(32))
        } else if normalized >= 0.0001f32 {
            (15.0f32, 0.001f32, 0.0001f32, u32::new(16))
        } else if normalized >= 0.00001f32 {
            (7.0f32, 0.0001f32, 0.00001f32, u32::new(8))
        } else if normalized >= 0.000001f32 {
            (3.0f32, 0.00001f32, 0.000001f32, u32::new(4))
        } else if normalized >= 0.0000001f32 {
            (1.0f32, 0.000001f32, 0.0000001f32, u32::new(2))
        } else {
            (0.0f32, 0.0000001f32, 0.00000001f32, u32::new(1))
        };

        let range = upper - lower;
        let safe_range = if range > 1e-10f32 {
            range
        } else {
            1.0f32.into()
        };
        let t = ((normalized - lower) / safe_range).clamp(0.0f32, 1.0f32);

        let fraction_f = (t * max_frac_f).round().clamp(0.0f32, 127.0f32);
        let fraction_raw = u32::cast_from(fraction_f);

        let max_frac_u = u32::cast_from(max_frac_f);
        let fraction = if fraction_raw > max_frac_u {
            max_frac_u
        } else {
            fraction_raw
        };

        indicator_val + fraction
    }
}

#[cube]
pub fn decode(code: u32) -> f32 {
    let is_zero = code == 0u32;

    if is_zero {
        f32::new(0.0)
    } else {
        // Find the indicator bit position. No sign bit to mask off — the
        // entire 8-bit value contributes to indicator + fraction.
        let (max_frac_f, upper, lower, indicator_val) = if code >= 128u32 {
            (127.0f32, 1.0f32, 0.1f32, u32::new(128))
        } else if code >= 64u32 {
            (63.0f32, 0.1f32, 0.01f32, u32::new(64))
        } else if code >= 32u32 {
            (31.0f32, 0.01f32, 0.001f32, u32::new(32))
        } else if code >= 16u32 {
            (15.0f32, 0.001f32, 0.0001f32, u32::new(16))
        } else if code >= 8u32 {
            (7.0f32, 0.0001f32, 0.00001f32, u32::new(8))
        } else if code >= 4u32 {
            (3.0f32, 0.00001f32, 0.000001f32, u32::new(4))
        } else if code >= 2u32 {
            (1.0f32, 0.000001f32, 0.0000001f32, u32::new(2))
        } else {
            (0.0f32, 0.0000001f32, 0.00000001f32, u32::new(1))
        };

        let fraction = code - indicator_val;
        let fraction_f = f32::cast_from(fraction);

        let safe_denom = if max_frac_f > 0.0f32 {
            max_frac_f
        } else {
            1.0f32.into()
        };
        let t = fraction_f / safe_denom;

        let range = upper - lower;
        lower + t * range
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::{bytes::Bytes, wgpu::WgpuRuntime};

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

    fn unsigned_roundtrip_via_kernel<R: Runtime>(
        client: &ComputeClient<R>,
        values: &[f32],
    ) -> (Vec<u32>, Vec<f32>) {
        let n = values.len();

        let input_bytes = f32::as_bytes(values).to_vec();
        let input_handle = client.create(Bytes::from_bytes_vec(input_bytes));

        let encoded_handle = client.empty(n * core::mem::size_of::<u32>());
        let decoded_handle = client.empty(n * core::mem::size_of::<f32>());

        let cube_dim = CubeDim::new(client, n);
        let units_per_cube = cube_dim.x * cube_dim.y * cube_dim.z;
        let cubes = (n as u32).div_ceil(units_per_cube);

        unsafe {
            roundtrip_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(cubes, 1, 1),
                cube_dim,
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
    fn test_unsigned_dynamic_quantization_roundtrip() {
        let client = TestRuntime::client(&Default::default());

        let values: Vec<f32> = vec![
            0.0, 0.5, -0.5, 1.0, -1.0, 0.1, -0.1, 0.01, -0.01, 0.001, 0.0001, 0.00001, 0.95, 0.05,
            0.005, 0.123, -0.456, 0.789,
        ];

        let (codes, decoded) = unsigned_roundtrip_via_kernel::<TestRuntime>(&client, &values);

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
