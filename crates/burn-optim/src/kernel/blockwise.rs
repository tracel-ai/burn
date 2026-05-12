use cubecl::prelude::*;

use crate::kernel::signed;
use crate::kernel::unsigned;
use crate::launch::PACK_SHIFT;
use crate::launch::PACKING_AMOUNT;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Scheme {
    Signed = 0,
    Unsigned = 1,
}

#[cube]
fn encode_dispatch(normalized: f32, #[comptime] scheme: u32) -> u32 {
    if comptime!(scheme == Scheme::Signed as u32) {
        signed::encode(normalized)
    } else if comptime!(scheme == Scheme::Unsigned as u32) {
        unsigned::encode(normalized)
    } else {
        comptime! { panic!("Scheme not a proper value."); }
    }
}

#[cube]
fn decode_dispatch(code: u32, #[comptime] scheme: u32) -> f32 {
    if comptime!(scheme == Scheme::Signed as u32) {
        signed::decode(code)
    } else if comptime!(scheme == Scheme::Unsigned as u32) {
        unsigned::decode(code)
    } else {
        comptime! { panic!("Scheme not a proper value."); }
    }
}

#[cube]
pub fn quantize_blockwise(
    shared: &SharedMemory<f32>,
    codes: &mut Array<u32>,
    scales: &Array<f32>,
    i: u32,
    shared_offset: u32,
    #[comptime] block_size: u32,
    #[comptime] scheme: u32,
) {
    let block = i / block_size;
    let scale = scales[block as usize];

    let v0 = shared[shared_offset as usize];
    let v1 = shared[shared_offset as usize + 1];
    let v2 = shared[shared_offset as usize + 2];
    let v3 = shared[shared_offset as usize + 3];

    let c0 = encode_dispatch(v0 / scale, scheme);
    let c1 = encode_dispatch(v1 / scale, scheme);
    let c2 = encode_dispatch(v2 / scale, scheme);
    let c3 = encode_dispatch(v3 / scale, scheme);

    let pack_idx = i / PACKING_AMOUNT;
    codes[pack_idx as usize] = c0 * PACK_SHIFT * PACK_SHIFT * PACK_SHIFT
        + c1 * PACK_SHIFT * PACK_SHIFT
        + c2 * PACK_SHIFT
        + c3;
}

#[cube]
pub fn dequantize_blockwise(
    packed: &Array<u32>,
    scales: &Array<f32>,
    i: u32,
    #[comptime] block_size: u32,
    #[comptime] scheme: u32,
) -> f32 {
    let pack_idx = i / PACKING_AMOUNT;
    let pack_pos = i % PACKING_AMOUNT;
    let packed_val = packed[pack_idx as usize];

    let code = if pack_pos == 0 {
        packed_val / (PACK_SHIFT * PACK_SHIFT * PACK_SHIFT)
    } else if pack_pos == 1 {
        (packed_val / (PACK_SHIFT * PACK_SHIFT)) % PACK_SHIFT
    } else if pack_pos == 2 {
        (packed_val / PACK_SHIFT) % PACK_SHIFT
    } else {
        packed_val % PACK_SHIFT
    };

    let normalized = decode_dispatch(code, scheme);
    let block = i / block_size;
    let scale = scales[block as usize];

    normalized * scale
}

#[cfg(test)]
mod blockwise_tests {
    use super::*;
    use cubecl::bytes::Bytes;
    use cubecl::cuda::CudaRuntime;
    use cubecl::hip::HipRuntime;
    use cubecl::prelude::*;
    use cubecl::wgpu::WgpuRuntime;

    // type TestRuntime = CudaRuntime;
    type TestRuntime = WgpuRuntime;
    // type TestRuntime = HipRuntime;

    const BLOCK_SIZE: u32 = 256;
    const N: usize = BLOCK_SIZE as usize;

    /// Quantize `input` into `codes`/`scales`, then dequantize back into `output`.
    /// Single block. Mirrors the staging-then-quantize pattern in `transform`.
    #[cube(launch_unchecked)]
    fn roundtrip_kernel(
        input: &Array<f32>,
        codes: &mut Array<u32>,
        scales: &mut Array<f32>,
        output: &mut Array<f32>,
        #[comptime] block_size: u32,
        #[comptime] plane_size: u32,
        #[comptime] scheme: u32,
    ) {
        let block = CUBE_POS_X;
        let unit = UNIT_POS_X;
        let block_start = block * block_size;
        let elements_per_thread = comptime!(block_size / plane_size);
        let quads_per_thread = comptime!(block_size / (plane_size * PACKING_AMOUNT));

        // Stage input into shared memory and compute per-block absmax for the scale.
        let mut shared = SharedMemory::<f32>::new(block_size as usize);
        let mut local_absmax = 0.0_f32;

        #[unroll]
        for iter in 0..elements_per_thread {
            let element = unit + iter * plane_size;
            let i = block_start + element;
            let v = input[i as usize];
            shared[element as usize] = v;
            local_absmax = max(local_absmax, v.abs());
        }

        let block_absmax = plane_max(local_absmax);
        let safe_scale = if block_absmax > 0.0_f32 {
            block_absmax
        } else {
            1.0_f32.into()
        };

        if unit == 0 {
            scales[block as usize] = safe_scale;
        }
        sync_cube();

        // Quantize: one quad per lane per iter.
        #[unroll]
        for iter in 0..quads_per_thread {
            let element = unit * PACKING_AMOUNT + iter * plane_size * PACKING_AMOUNT;
            let i = block_start + element;
            quantize_blockwise(&shared, codes, scales, i, element, block_size, scheme);
        }

        // Dequantize back, using the Pass-1-style stride.
        #[unroll]
        for iter in 0..elements_per_thread {
            let element = unit + iter * plane_size;
            let i = block_start + element;
            output[i as usize] = dequantize_blockwise(codes, scales, i, block_size, scheme);
        }
    }

    struct RoundtripResult {
        output: Vec<f32>,
        codes: Vec<u32>,
        scales: Vec<f32>,
    }

    fn run_roundtrip(input: &[f32], scheme: Scheme) -> RoundtripResult {
        let client = TestRuntime::client(&Default::default());
        let n = input.len();
        assert_eq!(
            n % BLOCK_SIZE as usize,
            0,
            "input length ({}) must be a multiple of BLOCK_SIZE ({})",
            n,
            BLOCK_SIZE,
        );
        let num_blocks = n as u32 / BLOCK_SIZE;
        let packed_count = n / PACKING_AMOUNT as usize;

        let input_h = client.create(Bytes::from_bytes_vec(f32::as_bytes(input).to_vec()));
        let codes_h = client.empty(packed_count * core::mem::size_of::<u32>());
        let scales_h = client.empty(num_blocks as usize * core::mem::size_of::<f32>());
        let output_h = client.empty(n * core::mem::size_of::<f32>());

        let plane_size = client.properties().hardware.plane_size_max;

        unsafe {
            roundtrip_kernel::launch_unchecked::<TestRuntime>(
                &client,
                CubeCount::Static(num_blocks, 1, 1),
                CubeDim::new(&client, plane_size as usize),
                ArrayArg::from_raw_parts(input_h, n),
                ArrayArg::from_raw_parts(codes_h.clone(), packed_count),
                ArrayArg::from_raw_parts(scales_h.clone(), num_blocks as usize),
                ArrayArg::from_raw_parts(output_h.clone(), n),
                BLOCK_SIZE,
                plane_size,
                scheme as u32,
            );
        }

        let read_f32 = |h| f32::from_bytes(&client.read_one_unchecked(h)).to_vec();
        let read_u32 = |h| u32::from_bytes(&client.read_one_unchecked(h)).to_vec();

        RoundtripResult {
            output: read_f32(output_h),
            codes: read_u32(codes_h.clone()),
            scales: read_f32(scales_h),
        }
    }

    /// Roundtrip tolerance: codes are quantized to a discrete grid, so error
    /// scales with the block's scale. 1/15 ~ 6.7% is generous for 4-bit grids;
    /// tighten if your grid is denser.
    fn assert_close(input: &[f32], output: &[f32], scale: f32) {
        let tol = scale / 15.0 + 1e-6;
        for (i, (&a, &b)) in input.iter().zip(output.iter()).enumerate() {
            let err = (a - b).abs();
            assert!(
                err <= tol,
                "lane {i}: input={a} output={b} err={err} tol={tol}",
            );
        }
    }

    #[test]
    fn roundtrip_signed_symmetric() {
        // Values in [-1, 1] at unit scale.
        let input: Vec<f32> = (0..N)
            .map(|i| {
                let t = (i as f32 / (N - 1) as f32) * 2.0 - 1.0;
                t * 0.95
            })
            .collect();
        let result = run_roundtrip(&input, Scheme::Signed);
        let scale = result.scales[0];
        assert!(scale > 0.0, "scale should be positive, got {}", scale);
        assert_close(&input, &result.output, scale);
    }

    #[test]
    fn roundtrip_unsigned_nonnegative() {
        let input: Vec<f32> = (0..N).map(|i| (i as f32 / N as f32) * 0.9).collect();
        let result = run_roundtrip(&input, Scheme::Unsigned);
        let scale = result.scales[0];
        assert!(scale > 0.0);
        assert_close(&input, &result.output, scale);
    }

    #[test]
    fn roundtrip_signed_alternating_sign() {
        // Stresses pack-position correctness: adjacent values have opposite signs,
        // so a swapped pack_pos would scramble the recovered ordering.
        let input: Vec<f32> = (0..N)
            .map(|i| if i % 2 == 0 { 0.7 } else { -0.7 })
            .collect();
        let result = run_roundtrip(&input, Scheme::Signed);

        for (i, &v) in result.output.iter().enumerate() {
            let expected_sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            assert!(
                v.signum() == expected_sign || v == 0.0,
                "lane {i}: expected sign {expected_sign}, got {v}",
            );
        }
    }

    #[test]
    fn roundtrip_signed_scaled() {
        // Large dynamic range — block absmax should drive the scale.
        let input: Vec<f32> = (0..N)
            .map(|i| {
                let t = (i as f32 / (N - 1) as f32) * 2.0 - 1.0;
                t * 100.0
            })
            .collect();
        let result = run_roundtrip(&input, Scheme::Signed);
        let scale = result.scales[0];
        assert!(
            (scale - 100.0).abs() < 1e-3,
            "scale should be ~100, got {}",
            scale,
        );
        assert_close(&input, &result.output, scale);
    }

    #[test]
    fn roundtrip_zero_input() {
        // Absmax is 0; safe_scale falls through to 1.0. Decoded values should be 0.
        let input = vec![0.0_f32; N];
        let result = run_roundtrip(&input, Scheme::Signed);
        assert_eq!(
            result.scales[0], 1.0,
            "safe_scale should fall through to 1.0"
        );
        for (i, &v) in result.output.iter().enumerate() {
            assert_eq!(v, 0.0, "lane {i}: expected 0, got {v}");
        }
    }
}
