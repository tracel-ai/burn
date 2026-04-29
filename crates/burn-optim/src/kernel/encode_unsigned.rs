//! CubeCL implementation of the unsigned dynamic encoding.

use cubecl::bytes::Bytes;
use cubecl::prelude::*;

#[cube]
pub fn encode_unsigned_one(normalized: f32) -> u32 {
    if normalized < 1e-7f32 {
        u32::new(0)
    } else {
        let (indicator, max_frac_f, upper, lower) = if normalized >= 0.1f32 {
            (u32::new(128), 127.0f32, 1.0f32, 0.1f32)
        } else if normalized >= 0.01f32 {
            (u32::new(64), 63.0f32, 0.1f32, 0.01f32)
        } else if normalized >= 0.001f32 {
            (u32::new(32), 31.0f32, 0.01f32, 0.001f32)
        } else if normalized >= 0.0001f32 {
            (u32::new(16), 15.0f32, 0.001f32, 0.0001f32)
        } else if normalized >= 0.00001f32 {
            (u32::new(8), 7.0f32, 0.0001f32, 0.00001f32)
        } else if normalized >= 0.000001f32 {
            (u32::new(4), 3.0f32, 0.00001f32, 0.000001f32)
        } else {
            (u32::new(2), 1.0f32, 0.000001f32, 0.0000001f32)
        };

        let range = upper - lower;
        let t = (normalized - lower) / range;
        let t_clamped = f32::clamp(t, 0.0f32, 1.0f32);

        let fraction_f = f32::round(t_clamped * max_frac_f);
        let fraction_clamped = f32::clamp(fraction_f, 0.0f32, max_frac_f);
        let fraction = fraction_clamped as u32;

        indicator | fraction
    }
}

#[cube(launch_unchecked)]
pub fn encode_unsigned_test_kernel(input: &Array<f32>, output: &mut Array<u32>) {
    if ABSOLUTE_POS < input.len() {
        output[ABSOLUTE_POS] = encode_unsigned_one(input[ABSOLUTE_POS]);
    }
}

pub fn encode_unsigned_via_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    input: &[f32],
) -> Vec<u32> {
    let n = input.len();

    let input_bytes_vec: Vec<u8> = f32::as_bytes(input).to_vec();
    let input_bytes = Bytes::from_bytes_vec(input_bytes_vec);

    let input_handle = client.create(input_bytes);
    let output_handle = client.empty(n * core::mem::size_of::<u32>());

    let threads_per_cube: u32 = 64;
    let cubes = (n as u32).div_ceil(threads_per_cube);

    let output_handle_for_launch = output_handle.clone();

    unsafe {
        encode_unsigned_test_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new(client, n),
            ArrayArg::from_raw_parts(input_handle, n),
            ArrayArg::from_raw_parts(output_handle_for_launch, n),
        );
    }

    let bytes: Bytes = client.read_one_unchecked(output_handle);
    let raw: &[u8] = &bytes;
    u32::from_bytes(raw).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_unsigned_one_cpu(normalized: f32) -> u32 {
        if normalized < 1e-7 {
            return 0;
        }

        let (indicator, max_frac_f, upper, lower): (u32, f32, f32, f32) = if normalized >= 0.1 {
            (128, 127.0, 1.0, 0.1)
        } else if normalized >= 0.01 {
            (64, 63.0, 0.1, 0.01)
        } else if normalized >= 0.001 {
            (32, 31.0, 0.01, 0.001)
        } else if normalized >= 0.0001 {
            (16, 15.0, 0.001, 0.0001)
        } else if normalized >= 0.00001 {
            (8, 7.0, 0.0001, 0.00001)
        } else if normalized >= 0.000001 {
            (4, 3.0, 0.00001, 0.000001)
        } else {
            (2, 1.0, 0.000001, 0.0000001)
        };

        let t = ((normalized - lower) / (upper - lower)).clamp(0.0, 1.0);
        let fraction = (t * max_frac_f).round().clamp(0.0, max_frac_f) as u32;
        indicator | fraction
    }

    #[cfg(feature = "test-cuda")]
    #[test]
    fn encode_unsigned_one_matches_reference_cuda() {
        use cubecl::cuda::CudaRuntime;
        run_match_test::<CudaRuntime>();
    }

    fn run_match_test<R: Runtime>() {
        let device = R::Device::default();
        let client = R::client(&device);

        let inputs: Vec<f32> = vec![
            0.0, 1e-8, -0.5, -1e-3, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 0.5, 0.123,
            0.789, 0.05, 0.005, 0.0005, 0.99, 0.999, 0.55, 0.055, 0.0055, 0.00055, 0.000055,
            0.0000055,
        ];

        let kernel_codes = encode_unsigned_via_kernel::<R>(&client, &inputs);
        let cpu_codes: Vec<u32> = inputs
            .iter()
            .copied()
            .map(encode_unsigned_one_cpu)
            .collect();

        for (i, (k, c)) in kernel_codes.iter().zip(cpu_codes.iter()).enumerate() {
            assert_eq!(
                k, c,
                "mismatch at index {i}: input = {}, kernel = {}, cpu = {}",
                inputs[i], k, c
            );
        }
    }
}
