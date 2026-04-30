//! Fused AdamW 8-bit step kernel.
//!
//! One workgroup per 256-element block. Within a workgroup:
//!   1. Load each thread's m_code, v_code from packed inputs.
//!   2. Decode to fp32 in registers using the per-block scales.
//!   3. Compute new m, new v via AdamW running averages (in registers).
//!   4. Compute delta = (m_new / (sqrt(v_use) + epsilon_eff)) * step_size.
//!   5. Workgroup absmax reduction over |m_new| -> new m scale.
//!   6. Workgroup max reduction over v_new (already >= 0) -> new v scale.
//!   7. Re-encode m_new, v_new with the new scales.
//!   8. Pack 4 codes per u32 and write to global memory.
//!
//! No AMSGrad. No cautious weight decay (caller does its decay step on the
//! returned delta, then subtracts from the parameter).
//!
//! Bias correction: the host caller precomputes `step_size = sqrt(1 - beta_2^t) / (1 - beta_1^t)`
//! and `epsilon_eff = epsilon * sqrt(1 - beta_2^t)`. The kernel uses these
//! directly so it doesn't need to know about t.

use crate::kernel::decode_signed::decode_signed_one;
use crate::kernel::decode_unsigned::decode_unsigned_one;
use crate::kernel::encode_signed::encode_signed_one;
use crate::kernel::encode_unsigned::encode_unsigned_one;
use cubecl::bytes::Bytes;
use cubecl::prelude::*;

const BLOCK_SIZE: usize = 256;
const PACK_FACTOR: usize = 4;
const PACKED_PER_BLOCK: usize = BLOCK_SIZE / PACK_FACTOR;

#[cube(launch_unchecked)]
pub fn fused_adamw8bit_step_kernel(
    grad: &Array<f32>,
    m_codes_in: &Array<u32>,
    m_scales_in: &Array<f32>,
    v_codes_in: &Array<u32>,
    v_scales_in: &Array<f32>,
    delta_out: &mut Array<f32>,
    m_codes_out: &mut Array<u32>,
    m_scales_out: &mut Array<f32>,
    v_codes_out: &mut Array<u32>,
    v_scales_out: &mut Array<f32>,
    beta_1: f32,
    beta_2: f32,
    epsilon_eff: f32,
    step_size: f32,
) {
    // Shared memory: two reduction buffers (m, v) plus two pack buffers.
    let mut shmem_m_abs = SharedMemory::<f32>::new(BLOCK_SIZE);
    let mut shmem_v = SharedMemory::<f32>::new(BLOCK_SIZE);
    let mut shmem_m_codes = SharedMemory::<u32>::new(BLOCK_SIZE);
    let mut shmem_v_codes = SharedMemory::<u32>::new(BLOCK_SIZE);

    let block_idx = CUBE_POS as usize;
    let tid = UNIT_POS as usize;
    let global_idx = block_idx * BLOCK_SIZE + tid;

    // ---- Step 1: load codes from packed format ----
    let pack_idx = block_idx * PACKED_PER_BLOCK + (tid / PACK_FACTOR);
    let slot = tid % PACK_FACTOR;
    let shift = (slot as u32) * 8u32;

    let m_packed = m_codes_in[pack_idx];
    let m_code = (m_packed >> shift) & 0xffu32;

    let v_packed = v_codes_in[pack_idx];
    let v_code = (v_packed >> shift) & 0xffu32;

    // ---- Step 2: decode to real-units fp32 ----
    let m_scale_old = m_scales_in[block_idx];
    let v_scale_old = v_scales_in[block_idx];
    let m_old = decode_signed_one(m_code) * m_scale_old;
    let v_old = decode_unsigned_one(v_code) * v_scale_old;

    // ---- Step 3: AdamW running averages ----
    let g = grad[global_idx];
    let factor_1 = 1.0f32 - beta_1;
    let factor_2 = 1.0f32 - beta_2;
    let m_new = beta_1 * m_old + factor_1 * g;
    let v_new = beta_2 * v_old + factor_2 * (g * g);

    // ---- Step 4: compute delta ----
    let denom = f32::sqrt(v_new) + epsilon_eff;
    let delta = (m_new / denom) * step_size;
    delta_out[global_idx] = delta;

    // ---- Step 5: m absmax reduction ----
    shmem_m_abs[tid] = f32::abs(m_new);
    shmem_v[tid] = v_new; // v_new is non-negative, no abs needed
    sync_cube();

    // Two reductions interleaved: same level for both, only one sync per level.
    if tid < 128usize {
        let m_other = shmem_m_abs[tid + 128usize];
        let m_mine = shmem_m_abs[tid];
        shmem_m_abs[tid] = f32::max(m_mine, m_other);

        let v_other = shmem_v[tid + 128usize];
        let v_mine = shmem_v[tid];
        shmem_v[tid] = f32::max(v_mine, v_other);
    }
    sync_cube();

    if tid < 64usize {
        let m_other = shmem_m_abs[tid + 64usize];
        let m_mine = shmem_m_abs[tid];
        shmem_m_abs[tid] = f32::max(m_mine, m_other);

        let v_other = shmem_v[tid + 64usize];
        let v_mine = shmem_v[tid];
        shmem_v[tid] = f32::max(v_mine, v_other);
    }
    sync_cube();

    if tid < 32usize {
        let m_other = shmem_m_abs[tid + 32usize];
        let m_mine = shmem_m_abs[tid];
        shmem_m_abs[tid] = f32::max(m_mine, m_other);

        let v_other = shmem_v[tid + 32usize];
        let v_mine = shmem_v[tid];
        shmem_v[tid] = f32::max(v_mine, v_other);
    }
    sync_cube();

    if tid < 16usize {
        let m_other = shmem_m_abs[tid + 16usize];
        let m_mine = shmem_m_abs[tid];
        shmem_m_abs[tid] = f32::max(m_mine, m_other);

        let v_other = shmem_v[tid + 16usize];
        let v_mine = shmem_v[tid];
        shmem_v[tid] = f32::max(v_mine, v_other);
    }
    sync_cube();

    if tid < 8usize {
        let m_other = shmem_m_abs[tid + 8usize];
        let m_mine = shmem_m_abs[tid];
        shmem_m_abs[tid] = f32::max(m_mine, m_other);

        let v_other = shmem_v[tid + 8usize];
        let v_mine = shmem_v[tid];
        shmem_v[tid] = f32::max(v_mine, v_other);
    }
    sync_cube();

    if tid < 4usize {
        let m_other = shmem_m_abs[tid + 4usize];
        let m_mine = shmem_m_abs[tid];
        shmem_m_abs[tid] = f32::max(m_mine, m_other);

        let v_other = shmem_v[tid + 4usize];
        let v_mine = shmem_v[tid];
        shmem_v[tid] = f32::max(v_mine, v_other);
    }
    sync_cube();

    if tid < 2usize {
        let m_other = shmem_m_abs[tid + 2usize];
        let m_mine = shmem_m_abs[tid];
        shmem_m_abs[tid] = f32::max(m_mine, m_other);

        let v_other = shmem_v[tid + 2usize];
        let v_mine = shmem_v[tid];
        shmem_v[tid] = f32::max(v_mine, v_other);
    }
    sync_cube();

    if tid < 1usize {
        let m_other = shmem_m_abs[tid + 1usize];
        let m_mine = shmem_m_abs[tid];
        shmem_m_abs[tid] = f32::max(m_mine, m_other);

        let v_other = shmem_v[tid + 1usize];
        let v_mine = shmem_v[tid];
        shmem_v[tid] = f32::max(v_mine, v_other);
    }
    sync_cube();

    let m_absmax_new = shmem_m_abs[0usize];
    let v_absmax_new = shmem_v[0usize];

    // ---- Step 6: write new scales (clamped to 1.0 for zero blocks) ----
    let m_scale_new = if m_absmax_new > 0.0f32 {
        m_absmax_new
    } else {
        f32::new(1.0)
    };
    let v_scale_new = if v_absmax_new > 0.0f32 {
        v_absmax_new
    } else {
        f32::new(1.0)
    };

    if tid == 0usize {
        m_scales_out[block_idx] = m_scale_new;
        v_scales_out[block_idx] = v_scale_new;
    }

    // ---- Step 7: re-encode and pack ----
    let m_code_new = encode_signed_one(m_new / m_scale_new);
    let v_code_new = encode_unsigned_one(v_new / v_scale_new);

    shmem_m_codes[tid] = m_code_new;
    shmem_v_codes[tid] = v_code_new;
    sync_cube();

    // Threads 0..64 each pack 4 m codes AND 4 v codes.
    if tid < PACKED_PER_BLOCK {
        let base = tid * PACK_FACTOR;

        let m0 = shmem_m_codes[base];
        let m1 = shmem_m_codes[base + 1usize];
        let m2 = shmem_m_codes[base + 2usize];
        let m3 = shmem_m_codes[base + 3usize];
        let m_packed_out = m0 | (m1 << 8u32) | (m2 << 16u32) | (m3 << 24u32);

        let v0 = shmem_v_codes[base];
        let v1 = shmem_v_codes[base + 1usize];
        let v2 = shmem_v_codes[base + 2usize];
        let v3 = shmem_v_codes[base + 3usize];
        let v_packed_out = v0 | (v1 << 8u32) | (v2 << 16u32) | (v3 << 24u32);

        let out_idx = block_idx * PACKED_PER_BLOCK + tid;
        m_codes_out[out_idx] = m_packed_out;
        v_codes_out[out_idx] = v_packed_out;
    }
}

/// Inputs to the fused step. The host pads grad to a multiple of BLOCK_SIZE
/// before calling. m and v state must already be in the matching layout.
pub struct FusedStepInput<'a> {
    pub grad: &'a [f32],
    pub m_codes: &'a [u32],
    pub m_scales: &'a [f32],
    pub v_codes: &'a [u32],
    pub v_scales: &'a [f32],
    pub original_len: usize,
    pub beta_1: f32,
    pub beta_2: f32,
    pub epsilon: f32,
    pub time_step: u32,
}

pub struct FusedStepOutput {
    /// Raw delta, same length as original_len (caller multiplies by lr).
    pub delta: Vec<f32>,
    pub m_codes_new: Vec<u32>,
    pub m_scales_new: Vec<f32>,
    pub v_codes_new: Vec<u32>,
    pub v_scales_new: Vec<f32>,
}

/// Run the fused step. Allocates fresh GPU buffers, copies in/out via Bytes.
/// This is the Stage 1 standalone primitive; production use plugs it into
/// the Burn optimizer in Stage 2.
pub fn fused_adamw8bit_step_via_kernel<R: Runtime>(
    client: &ComputeClient<R>,
    input: FusedStepInput,
) -> FusedStepOutput {
    let original_len = input.original_len;
    let padding = (BLOCK_SIZE - (original_len % BLOCK_SIZE)) % BLOCK_SIZE;
    let padded_len = original_len + padding;
    let num_blocks = padded_len / BLOCK_SIZE;

    debug_assert_eq!(input.grad.len(), original_len);
    debug_assert_eq!(input.m_codes.len(), num_blocks * PACKED_PER_BLOCK);
    debug_assert_eq!(input.m_scales.len(), num_blocks);
    debug_assert_eq!(input.v_codes.len(), num_blocks * PACKED_PER_BLOCK);
    debug_assert_eq!(input.v_scales.len(), num_blocks);

    // Pad grad with zeros.
    let mut grad_padded = Vec::with_capacity(padded_len);
    grad_padded.extend_from_slice(input.grad);
    grad_padded.resize(padded_len, 0.0f32);

    // Bias-correction-adjusted scalars passed to the kernel.
    // step_size = sqrt(1 - beta_2^t) / (1 - beta_1^t)
    // epsilon_eff = epsilon * sqrt(1 - beta_2^t)
    let t = input.time_step as i32;
    let correction1 = 1.0f32 - input.beta_1.powi(t);
    let correction2_sqrt = (1.0f32 - input.beta_2.powi(t)).sqrt();
    let step_size = correction2_sqrt / correction1;
    let epsilon_eff = input.epsilon * correction2_sqrt;

    // Upload inputs.
    let grad_handle = client.create(Bytes::from_bytes_vec(f32::as_bytes(&grad_padded).to_vec()));
    let m_codes_handle =
        client.create(Bytes::from_bytes_vec(u32::as_bytes(input.m_codes).to_vec()));
    let m_scales_handle = client.create(Bytes::from_bytes_vec(
        f32::as_bytes(input.m_scales).to_vec(),
    ));
    let v_codes_handle =
        client.create(Bytes::from_bytes_vec(u32::as_bytes(input.v_codes).to_vec()));
    let v_scales_handle = client.create(Bytes::from_bytes_vec(
        f32::as_bytes(input.v_scales).to_vec(),
    ));

    // Allocate outputs.
    let delta_handle = client.empty(padded_len * core::mem::size_of::<f32>());
    let m_codes_out_handle =
        client.empty(num_blocks * PACKED_PER_BLOCK * core::mem::size_of::<u32>());
    let m_scales_out_handle = client.empty(num_blocks * core::mem::size_of::<f32>());
    let v_codes_out_handle =
        client.empty(num_blocks * PACKED_PER_BLOCK * core::mem::size_of::<u32>());
    let v_scales_out_handle = client.empty(num_blocks * core::mem::size_of::<f32>());

    // Clone for readback (handles are consumed by ArrayArg::from_raw_parts).
    let delta_for_launch = delta_handle.clone();
    let m_codes_for_launch = m_codes_out_handle.clone();
    let m_scales_for_launch = m_scales_out_handle.clone();
    let v_codes_for_launch = v_codes_out_handle.clone();
    let v_scales_for_launch = v_scales_out_handle.clone();

    let cube_count = cubecl::CubeCount::Static(num_blocks as u32, 1, 1);
    let cube_dim = cubecl::CubeDim::new_1d(BLOCK_SIZE as u32);

    unsafe {
        fused_adamw8bit_step_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(grad_handle, padded_len),
            ArrayArg::from_raw_parts(m_codes_handle, input.m_codes.len()),
            ArrayArg::from_raw_parts(m_scales_handle, num_blocks),
            ArrayArg::from_raw_parts(v_codes_handle, input.v_codes.len()),
            ArrayArg::from_raw_parts(v_scales_handle, num_blocks),
            ArrayArg::from_raw_parts(delta_for_launch, padded_len),
            ArrayArg::from_raw_parts(m_codes_for_launch, num_blocks * PACKED_PER_BLOCK),
            ArrayArg::from_raw_parts(m_scales_for_launch, num_blocks),
            ArrayArg::from_raw_parts(v_codes_for_launch, num_blocks * PACKED_PER_BLOCK),
            ArrayArg::from_raw_parts(v_scales_for_launch, num_blocks),
            input.beta_1,
            input.beta_2,
            epsilon_eff,
            step_size,
        );
    }

    let delta_bytes: Bytes = client.read_one_unchecked(delta_handle);
    let m_codes_bytes: Bytes = client.read_one_unchecked(m_codes_out_handle);
    let m_scales_bytes: Bytes = client.read_one_unchecked(m_scales_out_handle);
    let v_codes_bytes: Bytes = client.read_one_unchecked(v_codes_out_handle);
    let v_scales_bytes: Bytes = client.read_one_unchecked(v_scales_out_handle);

    let mut delta = f32::from_bytes(&delta_bytes).to_vec();
    delta.truncate(original_len);

    FusedStepOutput {
        delta,
        m_codes_new: u32::from_bytes(&m_codes_bytes).to_vec(),
        m_scales_new: f32::from_bytes(&m_scales_bytes).to_vec(),
        v_codes_new: u32::from_bytes(&v_codes_bytes).to_vec(),
        v_scales_new: f32::from_bytes(&v_scales_bytes).to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::dequantize_blockwise_signed::dequantize_blockwise_signed_via_kernel;
    use crate::kernel::dequantize_blockwise_unsigned::dequantize_blockwise_unsigned_via_kernel;
    use crate::kernel::quantize_blockwise_signed::quantize_blockwise_signed_via_kernel;
    use crate::kernel::quantize_blockwise_unsigned::quantize_blockwise_unsigned_via_kernel;

    /// Pure-Rust reference for one fused step. Mirrors what
    /// `AdaptiveMomentumW8Bit::transform` does in the tensor-op impl, but
    /// over plain Vec<f32> rather than tensors so the test is self-contained.
    /// Returns (delta, m_new_fp32, v_new_fp32).
    fn cpu_reference_step(
        grad: &[f32],
        m_old_fp32: &[f32],
        v_old_fp32: &[f32],
        beta_1: f32,
        beta_2: f32,
        epsilon: f32,
        time_step: u32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = grad.len();
        assert_eq!(m_old_fp32.len(), n);
        assert_eq!(v_old_fp32.len(), n);

        let factor_1 = 1.0 - beta_1;
        let factor_2 = 1.0 - beta_2;

        let t = time_step as i32;
        let correction1 = 1.0_f32 - beta_1.powi(t);
        let correction2_sqrt = (1.0_f32 - beta_2.powi(t)).sqrt();
        let step_size = correction2_sqrt / correction1;
        let epsilon_eff = epsilon * correction2_sqrt;

        let mut delta = vec![0.0f32; n];
        let mut m_new = vec![0.0f32; n];
        let mut v_new = vec![0.0f32; n];

        for i in 0..n {
            m_new[i] = beta_1 * m_old_fp32[i] + factor_1 * grad[i];
            v_new[i] = beta_2 * v_old_fp32[i] + factor_2 * grad[i] * grad[i];
            let denom = v_new[i].sqrt() + epsilon_eff;
            delta[i] = (m_new[i] / denom) * step_size;
        }

        (delta, m_new, v_new)
    }

    #[cfg(feature = "test-cuda")]
    #[test]
    fn fused_step_matches_cpu_reference() {
        use cubecl::cuda::CudaRuntime;

        let device = <CudaRuntime as Runtime>::Device::default();
        let client = <CudaRuntime as Runtime>::client(&device);

        // 512 elements (2 blocks). Realistic-scale state: m ~ 0.01, v ~ 0.0001.
        let n = 512;
        let grad: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                (t * 2.0 - 1.0) * 0.5
            })
            .collect();
        let m_old_fp32: Vec<f32> = (0..n)
            .map(|i| ((i as f32 / (n - 1) as f32) * 2.0 - 1.0) * 0.01)
            .collect();
        let v_old_fp32: Vec<f32> = (0..n)
            .map(|i| (i as f32 / (n - 1) as f32) * 0.0001 + 1e-8)
            .collect();

        let beta_1 = 0.9f32;
        let beta_2 = 0.999f32;
        let epsilon = 1e-8f32;
        let time_step = 5u32;

        // Quantize the prior state via the existing blockwise kernels so we
        // feed the fused step inputs in exactly the layout it expects.
        let m_quant = quantize_blockwise_signed_via_kernel::<CudaRuntime>(&client, &m_old_fp32);
        let v_quant = quantize_blockwise_unsigned_via_kernel::<CudaRuntime>(&client, &v_old_fp32);

        // Sanity check on the state we're about to feed in: dequantizing it
        // should give back something close to m_old_fp32 / v_old_fp32. We use
        // *that* as the reference's m_old/v_old, since the kernel will see
        // the quantized version too. This isolates fused-step error from
        // quantization error in the comparison below.
        let m_old_dequant = dequantize_blockwise_signed_via_kernel::<CudaRuntime>(
            &client,
            &m_quant.codes,
            &m_quant.scales,
            n,
        );
        let v_old_dequant = dequantize_blockwise_unsigned_via_kernel::<CudaRuntime>(
            &client,
            &v_quant.codes,
            &v_quant.scales,
            n,
        );

        let (ref_delta, _ref_m_new, _ref_v_new) = cpu_reference_step(
            &grad,
            &m_old_dequant,
            &v_old_dequant,
            beta_1,
            beta_2,
            epsilon,
            time_step,
        );

        // Run the fused kernel.
        let fused = fused_adamw8bit_step_via_kernel::<CudaRuntime>(
            &client,
            FusedStepInput {
                grad: &grad,
                m_codes: &m_quant.codes,
                m_scales: &m_quant.scales,
                v_codes: &v_quant.codes,
                v_scales: &v_quant.scales,
                original_len: n,
                beta_1,
                beta_2,
                epsilon,
                time_step,
            },
        );

        // Compare delta element-wise. This is the per-element
        // pre-quantization output; we expect tight agreement.
        assert_eq!(fused.delta.len(), ref_delta.len());
        let mut max_rel = 0.0f32;
        let mut max_rel_idx = 0;
        for (i, (k, r)) in fused.delta.iter().zip(ref_delta.iter()).enumerate() {
            if r.abs() < 1e-6 {
                assert!(
                    k.abs() < 1e-4,
                    "kernel produced non-zero delta where reference is near zero at {i}: k={k}, r={r}"
                );
                continue;
            }
            let rel = (*k - *r).abs() / r.abs();
            if rel > max_rel {
                max_rel = rel;
                max_rel_idx = i;
            }
            assert!(
                rel < 0.05,
                "delta mismatch at index {i}: kernel = {k}, ref = {r}, rel = {rel}"
            );
        }
        println!(
            "Max delta relative error: {} at index {}",
            max_rel, max_rel_idx
        );

        // Now check the new state by dequantizing the fused output and
        // comparing to the reference m_new / v_new. The fused output went
        // through one extra round of quantization, so we expect the looser
        // ~5% tolerance from blockwise roundtrip.
        let m_new_recovered = dequantize_blockwise_signed_via_kernel::<CudaRuntime>(
            &client,
            &fused.m_codes_new,
            &fused.m_scales_new,
            n,
        );
        let v_new_recovered = dequantize_blockwise_unsigned_via_kernel::<CudaRuntime>(
            &client,
            &fused.v_codes_new,
            &fused.v_scales_new,
            n,
        );

        let (_d, ref_m_new_full, ref_v_new_full) = cpu_reference_step(
            &grad,
            &m_old_dequant,
            &v_old_dequant,
            beta_1,
            beta_2,
            epsilon,
            time_step,
        );

        // For state comparisons, use a looser tolerance — the kernel output
        // has gone through one quantization round-trip, which has ~5-10%
        // worst-case per-element error.
        for (i, (k, r)) in m_new_recovered
            .iter()
            .zip(ref_m_new_full.iter())
            .enumerate()
        {
            if r.abs() < 1e-5 {
                continue; // near-zero values get squashed to depth-6 and back
            }
            let rel = (*k - *r).abs() / r.abs();
            assert!(
                rel < 0.10,
                "m_new mismatch at index {i}: kernel = {k}, ref = {r}, rel = {rel}"
            );
        }

        for (i, (k, r)) in v_new_recovered
            .iter()
            .zip(ref_v_new_full.iter())
            .enumerate()
        {
            if r.abs() < 1e-7 {
                continue;
            }
            let rel = (*k - *r).abs() / r.abs();
            assert!(
                rel < 0.10,
                "v_new mismatch at index {i}: kernel = {k}, ref = {r}, rel = {rel}"
            );
        }
    }
}
