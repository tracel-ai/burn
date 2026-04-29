//! CubeCL implementation of the unsigned dynamic decoding.

use cubecl::bytes::Bytes;
use cubecl::prelude::*;

#[cube]
pub fn decode_unsigned_one(code: u32) -> f32 {
    if code == 0u32 {
        f32::new(0.0)
    } else {
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
        } else {
            (1.0f32, 0.000001f32, 0.0000001f32, u32::new(2))
        };

        let fraction = code - indicator_val;
        let fraction_f = fraction as f32;

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

#[cube(launch_unchecked)]
pub fn decode_unsigned_test_kernel(input: &Array<u32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = decode_unsigned_one(input[ABSOLUTE_POS]);
    }
}

pub fn decode_unsigned_via_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    codes: &[u32],
) -> Vec<f32> {
    let n = codes.len();

    let input_bytes_vec: Vec<u8> = u32::as_bytes(codes).to_vec();
    let input_bytes = Bytes::from_bytes_vec(input_bytes_vec);

    let input_handle = client.create(input_bytes);
    let output_handle = client.empty(n * core::mem::size_of::<f32>());

    let threads_per_cube: u32 = 64;
    let cubes = (n as u32).div_ceil(threads_per_cube);

    let output_handle_for_launch = output_handle.clone();

    unsafe {
        decode_unsigned_test_kernel::launch_unchecked::<R>(
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
    use crate::kernel::encode_unsigned::encode_unsigned_via_kernel;

    fn decode_unsigned_one_cpu(code: u32) -> f32 {
        if code == 0 {
            return 0.0;
        }

        let (max_frac_f, upper, lower_f, indicator_val): (f32, f32, f32, u32) = if code >= 128 {
            (127.0, 1.0, 0.1, 128)
        } else if code >= 64 {
            (63.0, 0.1, 0.01, 64)
        } else if code >= 32 {
            (31.0, 0.01, 0.001, 32)
        } else if code >= 16 {
            (15.0, 0.001, 0.0001, 16)
        } else if code >= 8 {
            (7.0, 0.0001, 0.00001, 8)
        } else if code >= 4 {
            (3.0, 0.00001, 0.000001, 4)
        } else {
            (1.0, 0.000001, 0.0000001, 2)
        };

        let fraction = (code - indicator_val) as f32;
        let safe_denom = if max_frac_f > 0.0 { max_frac_f } else { 1.0 };
        let t = fraction / safe_denom;
        lower_f + t * (upper - lower_f)
    }

    #[cfg(feature = "test-cuda")]
    #[test]
    fn decode_unsigned_one_matches_reference_cuda() {
        use cubecl::cuda::CudaRuntime;
        run_match_test::<CudaRuntime>();
    }

    #[cfg(feature = "test-cuda")]
    #[test]
    fn unsigned_roundtrip_cuda() {
        use cubecl::cuda::CudaRuntime;
        run_roundtrip_test::<CudaRuntime>();
    }

    fn run_match_test<R: Runtime>() {
        let device = R::Device::default();
        let client = R::client(&device);

        let inputs: Vec<u32> = (0u32..256u32).collect();

        let kernel_decoded = decode_unsigned_via_kernel::<R>(&client, &inputs);
        let cpu_decoded: Vec<f32> = inputs
            .iter()
            .copied()
            .map(decode_unsigned_one_cpu)
            .collect();

        for (i, (k, c)) in kernel_decoded.iter().zip(cpu_decoded.iter()).enumerate() {
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

        let inputs: Vec<f32> = vec![
            0.0, 1e-8, 5e-5, 5e-4, 7.5e-4, 5e-3, 2.5e-3, 5e-2, 7.5e-2, 0.5, 0.123, 0.789, 0.05,
            0.005, 0.99, 0.25, 0.75,
        ];

        let codes = encode_unsigned_via_kernel::<R>(&client, &inputs);
        let recovered = decode_unsigned_via_kernel::<R>(&client, &codes);

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
                rel < 0.3,
                "roundtrip rel error too high at index {i}: orig = {orig}, rec = {rec}, rel = {rel}"
            );
        }
    }
}
