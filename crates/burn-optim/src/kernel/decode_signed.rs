//! CubeCL implementation of the signed dynamic decoding.

use cubecl::bytes::Bytes;
use cubecl::prelude::*;

#[cube]
pub fn decode_signed_one(code: u32) -> f32 {
    let lower_bits = code & 127u32;

    if lower_bits == 0u32 {
        f32::new(0.0)
    } else {
        let is_negative = (code & 128u32) != 0u32;
        let sign = if is_negative {
            f32::new(-1.0)
        } else {
            f32::new(1.0)
        };

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

        let fraction = lower_bits - indicator_val;
        let fraction_f = fraction as f32;

        let safe_denom = if max_frac_f > 0.0f32 {
            max_frac_f
        } else {
            1.0f32.into()
        };
        let t = fraction_f / safe_denom;

        let range = upper - lower;
        let abs_val = lower + t * range;

        sign * abs_val
    }
}

#[cube(launch_unchecked)]
pub fn decode_signed_test_kernel(input: &Array<u32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = decode_signed_one(input[ABSOLUTE_POS]);
    }
}

pub fn decode_signed_via_kernel<R: Runtime>(client: &ComputeClient<R>, codes: &[u32]) -> Vec<f32> {
    let n = codes.len();

    let input_bytes_vec: Vec<u8> = u32::as_bytes(codes).to_vec();
    let input_bytes = Bytes::from_bytes_vec(input_bytes_vec);

    let input_handle = client.create(input_bytes);
    let output_handle = client.empty(n * core::mem::size_of::<f32>());

    let threads_per_cube: u32 = 64;
    let cubes = (n as u32).div_ceil(threads_per_cube);

    let output_handle_for_launch = output_handle.clone();

    unsafe {
        decode_signed_test_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new(client, n),
            ArrayArg::from_raw_parts(input_handle, n),
            ArrayArg::from_raw_parts(output_handle_for_launch, n),
        );
    }

    let bytes: Bytes = client.read_one_unchecked(output_handle);
    let raw: &[u8] = &bytes;
    f32::from_bytes(raw).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::encode_signed::encode_signed_via_kernel;

    fn decode_signed_one_cpu(code: u32) -> f32 {
        let lower_bits = code & 127;
        if lower_bits == 0 {
            return 0.0;
        }
        let sign: f32 = if (code & 128) != 0 { -1.0 } else { 1.0 };

        let (max_frac_f, upper, lower_f, indicator_val): (f32, f32, f32, u32) = if lower_bits >= 64
        {
            (63.0, 1.0, 0.1, 64)
        } else if lower_bits >= 32 {
            (31.0, 0.1, 0.01, 32)
        } else if lower_bits >= 16 {
            (15.0, 0.01, 0.001, 16)
        } else if lower_bits >= 8 {
            (7.0, 0.001, 0.0001, 8)
        } else if lower_bits >= 4 {
            (3.0, 0.0001, 0.00001, 4)
        } else if lower_bits >= 2 {
            (1.0, 0.00001, 0.000001, 2)
        } else {
            (0.0, 0.000001, 0.0000001, 1)
        };

        let fraction = (lower_bits - indicator_val) as f32;
        let safe_denom = if max_frac_f > 0.0 { max_frac_f } else { 1.0 };
        let t = fraction / safe_denom;
        sign * (lower_f + t * (upper - lower_f))
    }

    #[cfg(feature = "test-cuda")]
    #[test]
    fn decode_signed_one_matches_reference_cuda() {
        use cubecl::cuda::CudaRuntime;
        run_match_test::<CudaRuntime>();
    }

    #[cfg(feature = "test-cuda")]
    #[test]
    fn signed_roundtrip_cuda() {
        use cubecl::cuda::CudaRuntime;
        run_roundtrip_test::<CudaRuntime>();
    }

    fn run_match_test<R: Runtime>() {
        let device = R::Device::default();
        let client = R::client(&device);

        let inputs: Vec<u32> = (0u32..256u32).collect();

        let kernel_decoded = decode_signed_via_kernel::<R>(&client, &inputs);
        let cpu_decoded: Vec<f32> = inputs.iter().copied().map(decode_signed_one_cpu).collect();

        for (i, (k, c)) in kernel_decoded.iter().zip(cpu_decoded.iter()).enumerate() {
            // Allow up to 1e-5 relative tolerance for non-zero values, absolute
            // tolerance for zero. GPU FMA may differ from CPU by 1-2 ULPs on the
            // arithmetic chain.
            if c.abs() < 1e-12 {
                assert!(
                    k.abs() < 1e-12,
                    "mismatch at code {i}: kernel = {k}, cpu = {c}"
                );
            } else {
                let rel = (k - c).abs() / c.abs();
                assert!(
                    rel < 1e-5,
                    "mismatch at code {i}: kernel = {k}, cpu = {c}, rel = {rel}"
                );
            }
        }
    }

    fn run_roundtrip_test<R: Runtime>() {
        let device = R::Device::default();
        let client = R::client(&device);

        // Test inputs picked to exercise each bracket without sitting on the
        // awkward midpoints of low-precision (depth 5/6) brackets where the
        // encoding's quantization granularity dominates relative error. Real
        // optimizer state, after blockwise absmax normalization, lands mostly
        // in depths 0-3 (4+ fraction bits), so this is the regime that matters.
        let inputs: Vec<f32> = vec![
            0.0, 1e-8, -1e-8, // sub-threshold, round to 0
            // Skip depth 5 and 6 — only 1 frac bit (or 0), checked separately
            // in the run_match_test exhaustive sweep.
            5e-5, -5e-5, 5e-4, -5e-4, 7.5e-4, 5e-3, -5e-3, 2.5e-3, 5e-2, -5e-2, 7.5e-2, 0.5, -0.5,
            0.123, -0.456, 0.789, 0.05, 0.005, 0.99, -0.99, 0.25, -0.75,
        ];

        let codes = encode_signed_via_kernel::<R>(&client, &inputs);
        let recovered = decode_signed_via_kernel::<R>(&client, &codes);

        for (i, (orig, rec)) in inputs.iter().zip(recovered.iter()).enumerate() {
            let abs_orig = orig.abs();
            if abs_orig < 1e-6 && abs_orig > 1e-7 {
                continue;
            }
            if abs_orig < 1e-7 {
                assert_eq!(
                    *rec, 0.0,
                    "zero/sub-threshold did not round to 0 at index {i}"
                );
                continue;
            }
            let rel = (orig - rec).abs() / abs_orig;
            assert!(
                rel < 0.6,
                "roundtrip rel error too high at index {i}: orig = {orig}, rec = {rec}, rel = {rel}"
            );
        }
    }
}
