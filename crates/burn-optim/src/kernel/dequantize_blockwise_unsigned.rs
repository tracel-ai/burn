//! CubeCL implementation of blockwise unsigned dynamic dequantization.

use crate::kernel::decode_unsigned::decode_unsigned_one;
use cubecl::bytes::Bytes;
use cubecl::prelude::*;

const BLOCK_SIZE: usize = 256;
const PACK_FACTOR: usize = 4;
const PACKED_PER_BLOCK: usize = BLOCK_SIZE / PACK_FACTOR;

#[cube(launch_unchecked)]
pub fn dequantize_blockwise_unsigned_kernel(
    codes_in: &Array<u32>,
    scales_in: &Array<f32>,
    output: &mut Array<f32>,
) {
    let block_idx = CUBE_POS as usize;
    let tid = UNIT_POS as usize;
    let global_idx = block_idx * BLOCK_SIZE + tid;

    let pack_idx = block_idx * PACKED_PER_BLOCK + (tid / PACK_FACTOR);
    let slot = tid % PACK_FACTOR;

    let packed = codes_in[pack_idx];
    let shift = (slot as u32) * 8u32;
    let code = (packed >> shift) & 0xffu32;

    let scale = scales_in[block_idx];
    let normalized = decode_unsigned_one(code);
    output[global_idx] = normalized * scale;
}

pub fn dequantize_blockwise_unsigned_via_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    codes: &[u32],
    scales: &[f32],
    original_len: usize,
) -> Vec<f32> {
    let num_blocks = scales.len();
    let padded_len = num_blocks * BLOCK_SIZE;
    debug_assert_eq!(codes.len(), num_blocks * PACKED_PER_BLOCK);

    let codes_bytes = Bytes::from_bytes_vec(u32::as_bytes(codes).to_vec());
    let scales_bytes = Bytes::from_bytes_vec(f32::as_bytes(scales).to_vec());

    let codes_handle = client.create(codes_bytes);
    let scales_handle = client.create(scales_bytes);
    let output_handle = client.empty(padded_len * core::mem::size_of::<f32>());

    let output_handle_for_launch = output_handle.clone();

    let cube_count = cubecl::CubeCount::Static(num_blocks as u32, 1, 1);
    let cube_dim = cubecl::CubeDim::new_1d(BLOCK_SIZE as u32);

    unsafe {
        dequantize_blockwise_unsigned_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(codes_handle, codes.len()),
            ArrayArg::from_raw_parts(scales_handle, num_blocks),
            ArrayArg::from_raw_parts(output_handle_for_launch, padded_len),
        );
    }

    let output_bytes: Bytes = client.read_one_unchecked(output_handle);
    let mut out = f32::from_bytes(&output_bytes).to_vec();
    out.truncate(original_len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::quantize_blockwise_unsigned::quantize_blockwise_unsigned_via_kernel;

    #[cfg(feature = "test-cuda")]
    #[test]
    fn dequantize_blockwise_unsigned_roundtrip() {
        use cubecl::cuda::CudaRuntime;

        let device = <CudaRuntime as Runtime>::Device::default();
        let client = <CudaRuntime as Runtime>::client(&device);

        let n = 512;
        let data: Vec<f32> = (0..n)
            .map(|i| (i as f32 / (n - 1) as f32) * 0.9 + 0.001)
            .collect();

        let q = quantize_blockwise_unsigned_via_kernel::<CudaRuntime>(&client, &data);
        let recovered = dequantize_blockwise_unsigned_via_kernel::<CudaRuntime>(
            &client,
            &q.codes,
            &q.scales,
            q.original_len,
        );

        assert_eq!(recovered.len(), data.len());

        for (i, (orig, rec)) in data.iter().zip(recovered.iter()).enumerate() {
            if orig.abs() < 1e-6 {
                assert!(
                    rec.abs() < 1e-5,
                    "near-zero input did not stay near zero at index {i}: orig = {orig}, rec = {rec}"
                );
                continue;
            }
            // Unsigned has one extra fraction bit per bracket vs signed,
            // so error should be tighter.
            let rel = (orig - rec).abs() / orig.abs();
            assert!(
                rel < 0.05,
                "roundtrip rel error too high at index {i}: orig = {orig}, rec = {rec}, rel = {rel}"
            );
        }
    }

    #[cfg(feature = "test-cuda")]
    #[test]
    fn dequantize_blockwise_unsigned_matches_reference() {
        use crate::quantization::{
            blockwise::{dequantize_blockwise, quantize_blockwise},
            unsigned_dynamic,
        };
        use burn_core::tensor::Tensor;
        use burn_ndarray::NdArray;
        use cubecl::cuda::CudaRuntime;

        let device = <CudaRuntime as Runtime>::Device::default();
        let client = <CudaRuntime as Runtime>::client(&device);

        let n = 512;
        let data: Vec<f32> = (0..n)
            .map(|i| (i as f32 / (n - 1) as f32) * 0.9 + 0.001)
            .collect();

        let q = quantize_blockwise_unsigned_via_kernel::<CudaRuntime>(&client, &data);
        let kernel_recovered = dequantize_blockwise_unsigned_via_kernel::<CudaRuntime>(
            &client,
            &q.codes,
            &q.scales,
            q.original_len,
        );

        let nd_device = Default::default();
        let input_tensor =
            Tensor::<NdArray, 1>::from_floats(data.as_slice(), &nd_device).reshape([2, BLOCK_SIZE]);
        let q_ref = quantize_blockwise(input_tensor, BLOCK_SIZE, unsigned_dynamic::encode);
        let dequant_ref = dequantize_blockwise(q_ref, BLOCK_SIZE, unsigned_dynamic::decode);
        let ref_recovered: Vec<f32> = dequant_ref.into_data().to_vec().unwrap();

        assert_eq!(kernel_recovered.len(), ref_recovered.len());
        for (i, (k, r)) in kernel_recovered
            .iter()
            .zip(ref_recovered.iter())
            .enumerate()
        {
            if r.abs() < 1e-6 {
                assert!(
                    k.abs() < 1e-5,
                    "kernel produced non-zero where reference is near-zero at {i}: k = {k}, r = {r}"
                );
                continue;
            }
            let rel = (*k - *r).abs() / r.abs();
            assert!(
                rel < 0.05,
                "kernel/reference mismatch at index {i}: kernel = {k}, ref = {r}, rel = {rel}"
            );
        }
    }
}
