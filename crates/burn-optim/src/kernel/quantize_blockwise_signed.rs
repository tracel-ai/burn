//! CubeCL implementation of blockwise signed dynamic quantization.

use crate::kernel::encode_signed::encode_signed_one;
use cubecl::bytes::Bytes;
use cubecl::prelude::*;

const BLOCK_SIZE: usize = 256;
const PACK_FACTOR: usize = 4;
const PACKED_PER_BLOCK: usize = BLOCK_SIZE / PACK_FACTOR; // 64

#[cube(launch_unchecked)]
pub fn quantize_blockwise_signed_kernel(
    input: &Array<f32>,
    codes_out: &mut Array<u32>,
    scales_out: &mut Array<f32>,
) {
    let mut absmax_buf = SharedMemory::<f32>::new(BLOCK_SIZE);
    let mut code_buf = SharedMemory::<u32>::new(BLOCK_SIZE);

    let block_idx = CUBE_POS as usize;
    let tid = UNIT_POS as usize;
    let global_idx = block_idx * BLOCK_SIZE + tid;

    let x = input[global_idx];

    // ---- Step 1: load |x| into shared memory ----
    absmax_buf[tid] = f32::abs(x);
    sync_cube();

    // ---- Step 2: tree reduction for max ----
    if tid < 128usize {
        let other = absmax_buf[tid + 128usize];
        let mine = absmax_buf[tid];
        absmax_buf[tid] = f32::max(mine, other);
    }
    sync_cube();

    if tid < 64usize {
        let other = absmax_buf[tid + 64usize];
        let mine = absmax_buf[tid];
        absmax_buf[tid] = f32::max(mine, other);
    }
    sync_cube();

    if tid < 32usize {
        let other = absmax_buf[tid + 32usize];
        let mine = absmax_buf[tid];
        absmax_buf[tid] = f32::max(mine, other);
    }
    sync_cube();

    if tid < 16usize {
        let other = absmax_buf[tid + 16usize];
        let mine = absmax_buf[tid];
        absmax_buf[tid] = f32::max(mine, other);
    }
    sync_cube();

    if tid < 8usize {
        let other = absmax_buf[tid + 8usize];
        let mine = absmax_buf[tid];
        absmax_buf[tid] = f32::max(mine, other);
    }
    sync_cube();

    if tid < 4usize {
        let other = absmax_buf[tid + 4usize];
        let mine = absmax_buf[tid];
        absmax_buf[tid] = f32::max(mine, other);
    }
    sync_cube();

    if tid < 2usize {
        let other = absmax_buf[tid + 2usize];
        let mine = absmax_buf[tid];
        absmax_buf[tid] = f32::max(mine, other);
    }
    sync_cube();

    if tid < 1usize {
        let other = absmax_buf[tid + 1usize];
        let mine = absmax_buf[tid];
        absmax_buf[tid] = f32::max(mine, other);
    }
    sync_cube();

    let absmax = absmax_buf[0usize];

    // ---- Step 3: thread 0 writes the scale ----
    let scale = if absmax > 0.0f32 {
        absmax
    } else {
        f32::new(1.0)
    };
    if tid == 0usize {
        scales_out[block_idx] = scale;
    }

    // ---- Step 4: normalize and encode ----
    let normalized = x / scale;
    let code = encode_signed_one(normalized);

    // ---- Step 5: write code into shared buffer for packing ----
    code_buf[tid] = code;
    sync_cube();

    // ---- Step 6: threads 0..64 pack four codes each into one u32 ----
    if tid < PACKED_PER_BLOCK {
        let base = tid * PACK_FACTOR;
        let c0 = code_buf[base];
        let c1 = code_buf[base + 1usize];
        let c2 = code_buf[base + 2usize];
        let c3 = code_buf[base + 3usize];

        let packed = c0 | (c1 << 8u32) | (c2 << 16u32) | (c3 << 24u32);

        let out_idx = block_idx * PACKED_PER_BLOCK + tid;
        codes_out[out_idx] = packed;
    }
}

pub struct QuantizedBlockwise {
    pub codes: Vec<u32>,
    pub scales: Vec<f32>,
    pub original_len: usize,
    pub padded_len: usize,
}

pub fn quantize_blockwise_signed_via_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    input: &[f32],
) -> QuantizedBlockwise {
    let original_len = input.len();
    let padding = (BLOCK_SIZE - (original_len % BLOCK_SIZE)) % BLOCK_SIZE;
    let padded_len = original_len + padding;

    let mut padded = Vec::with_capacity(padded_len);
    padded.extend_from_slice(input);
    padded.resize(padded_len, 0.0f32);

    let num_blocks = padded_len / BLOCK_SIZE;
    let codes_len = num_blocks * PACKED_PER_BLOCK;

    let input_bytes = Bytes::from_bytes_vec(f32::as_bytes(&padded).to_vec());
    let input_handle = client.create(input_bytes);

    let codes_handle = client.empty(codes_len * core::mem::size_of::<u32>());
    let scales_handle = client.empty(num_blocks * core::mem::size_of::<f32>());

    let codes_handle_for_launch = codes_handle.clone();
    let scales_handle_for_launch = scales_handle.clone();

    let cube_count = cubecl::CubeCount::Static(num_blocks as u32, 1, 1);
    let cube_dim = cubecl::CubeDim::new_1d(BLOCK_SIZE as u32);

    unsafe {
        quantize_blockwise_signed_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(input_handle, padded_len),
            ArrayArg::from_raw_parts(codes_handle_for_launch, codes_len),
            ArrayArg::from_raw_parts(scales_handle_for_launch, num_blocks),
        );
    }

    let codes_bytes: Bytes = client.read_one_unchecked(codes_handle);
    let scales_bytes: Bytes = client.read_one_unchecked(scales_handle);

    let codes = u32::from_bytes(&codes_bytes).to_vec();
    let scales = f32::from_bytes(&scales_bytes).to_vec();

    QuantizedBlockwise {
        codes,
        scales,
        original_len,
        padded_len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unpack_codes(packed: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(packed.len() * 4);
        for word in packed {
            out.push((*word & 0xff) as u8);
            out.push(((*word >> 8) & 0xff) as u8);
            out.push(((*word >> 16) & 0xff) as u8);
            out.push(((*word >> 24) & 0xff) as u8);
        }
        out
    }

    #[cfg(feature = "test-cuda")]
    #[test]
    fn quantize_blockwise_signed_matches_reference() {
        use crate::quantization::{blockwise::quantize_blockwise, signed_dynamic};
        use burn_core::tensor::Tensor;
        use burn_ndarray::NdArray;
        use cubecl::cuda::CudaRuntime;

        let device = <CudaRuntime as Runtime>::Device::default();
        let client = <CudaRuntime as Runtime>::client(&device);

        let n = 512;
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                (t * 2.0 - 1.0) * 0.9
            })
            .collect();

        let kernel_result = quantize_blockwise_signed_via_kernel::<CudaRuntime>(&client, &data);
        let kernel_codes_unpacked = unpack_codes(&kernel_result.codes);

        let nd_device = Default::default();
        let input_tensor =
            Tensor::<NdArray, 1>::from_floats(data.as_slice(), &nd_device).reshape([2, BLOCK_SIZE]);
        let reference = quantize_blockwise(input_tensor, BLOCK_SIZE, signed_dynamic::encode);
        let ref_scales: Vec<f32> = reference.scales.into_data().to_vec().unwrap();

        assert_eq!(kernel_result.scales.len(), ref_scales.len());
        for (i, (k, r)) in kernel_result
            .scales
            .iter()
            .zip(ref_scales.iter())
            .enumerate()
        {
            let rel = (*k - *r).abs() / (*r).abs().max(1e-12);
            assert!(
                rel < 1e-5,
                "scale mismatch at block {i}: kernel = {k}, reference = {r}"
            );
        }

        let ref_codes_packed: Vec<i64> = reference.quantized.into_data().to_vec().unwrap();
        let mut ref_codes_unpacked = Vec::with_capacity(ref_codes_packed.len() * 2);
        for word in &ref_codes_packed {
            ref_codes_unpacked.push((*word / 256) as u8);
            ref_codes_unpacked.push((*word % 256) as u8);
        }

        assert_eq!(kernel_codes_unpacked.len(), ref_codes_unpacked.len());
        for (i, (k, r)) in kernel_codes_unpacked
            .iter()
            .zip(ref_codes_unpacked.iter())
            .enumerate()
        {
            let diff = (*k as i32 - *r as i32).abs();
            assert!(
                diff <= 1,
                "code mismatch at index {i}: kernel = {k}, reference = {r}"
            );
        }
    }
}
