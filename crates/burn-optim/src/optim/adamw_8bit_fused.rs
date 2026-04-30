//! Fused 8-bit AdamW optimizer.
//!
//! This is the GPU-resident version of [`super::adamw_8bit::AdamW8Bit`]: the
//! per-step decode → AdamW math → re-encode pipeline runs as a single fused
//! CubeCL kernel, with no intermediate fp32 tensors materialized in global
//! memory. See `crate::kernel::fused_adamw8bit_step` for the kernel itself.
//!
//! Backend constraint: `CubeBackend<R>` only. Runs on every cubecl-backed
//! device (CUDA, Metal, ROCm, WGPU, Vulkan). For NdArray / LibTorch / Candle
//! backends, use `super::adamw_8bit::AdamW8Bit` (tensor-op reference impl)
//! instead.
//!
//! State layout per parameter tensor:
//!   - `moment_1.codes`: u32-packed (4 codes per word), padded length / 4
//!   - `moment_1.scales`: f32, one per 256-element block
//!   - `moment_2.codes`, `moment_2.scales`: same layout, unsigned encoding
//!   - `original_shape`: the unpadded param shape, kept for the delta's reshape
//!
//! The kernel's `delta` output is a flat fp32 array of padded length; we
//! truncate to the param's element count and reshape before returning.

use burn_core as burn;

use burn::config::Config;
use burn::module::AutodiffModule;
use burn::record::Record;
use burn::tensor::{
    Int, Shape, Tensor, TensorPrimitive,
    backend::{AutodiffBackend, Backend},
    ops::Device,
};
use burn_core::tensor::DType;
use burn_cubecl::{CubeBackend, CubeRuntime, tensor::CubeTensor};
use cubecl::CubeElement;
use cubecl::Metadata;
use cubecl::client::ComputeClient;
use cubecl::server::Handle;

use crate::kernel::fused_adamw8bit_step::fused_adamw8bit_step_kernel;
use crate::optim::{SimpleOptimizer, adaptor::OptimizerAdaptor};
use crate::{LearningRate, grad_clipping::GradientClippingConfig};

const BLOCK_SIZE: usize = 256;
const PACK_FACTOR: usize = 4;
const PACKED_PER_BLOCK: usize = BLOCK_SIZE / PACK_FACTOR;

/// Configuration for the fused 8-bit AdamW optimizer.
///
/// Mirrors [`super::adamw_8bit::AdamWConfig8Bit`] but the optimizer it
/// initializes is constrained to CubeCL backends.
#[derive(Config, Debug)]
pub struct AdamWConfig8BitFused {
    #[config(default = 0.9)]
    beta_1: f32,
    #[config(default = 0.999)]
    beta_2: f32,
    #[config(default = 1e-8)]
    epsilon: f32,
    #[config(default = 1e-4)]
    weight_decay: f32,
    grad_clipping: Option<GradientClippingConfig>,
}

#[derive(Clone)]
pub struct AdamW8BitFused {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    weight_decay: f32,
}

#[derive(Record, Clone)]
pub struct AdamW8BitFusedState<B: Backend, const D: usize> {
    pub time: usize,
    pub original_shape: [usize; D],
    pub moment_1_codes: Tensor<B, 1, Int>,
    pub moment_1_scales: Tensor<B, 1>,
    pub moment_2_codes: Tensor<B, 1, Int>,
    pub moment_2_scales: Tensor<B, 1>,
}

impl<R: CubeRuntime> SimpleOptimizer<CubeBackend<R>> for AdamW8BitFused {
    type State<const D: usize> = AdamW8BitFusedState<CubeBackend<R>, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<CubeBackend<R>, D>,
        grad: Tensor<CubeBackend<R>, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<CubeBackend<R>, D>, Option<Self::State<D>>) {
        let original_shape = tensor.shape().dims::<D>();
        let total_elements: usize = original_shape.iter().product();
        let padding = (BLOCK_SIZE - (total_elements % BLOCK_SIZE)) % BLOCK_SIZE;
        let padded_len = total_elements + padding;
        let num_blocks = padded_len / BLOCK_SIZE;

        // Get the underlying CubeTensor for grad. We'll pull the client off it
        // and use that for all kernel launches in this step.
        let grad_primitive = grad.into_primitive().tensor();
        let client = grad_primitive.client.clone();
        let device = grad_primitive.device.clone();

        // Pad grad to a multiple of BLOCK_SIZE. Easiest: read into host, pad,
        // upload. This is a perf cost we'll optimize later by either
        // requiring multiples-of-256 shapes or by padding on-device.
        let grad_handle = pad_to_block_size::<R>(&client, &grad_primitive, padded_len);

        // Initialize state on first step if absent. Codes start at 0 (which
        // decodes to 0 with any scale), scales start at 1.0 (so dequant of
        // zero codes gives zero, and so the kernel's normalize-by-scale step
        // is well-defined).
        let (m_codes_handle, m_scales_handle, v_codes_handle, v_scales_handle, time) = match &state
        {
            Some(s) => {
                let m_codes = s.moment_1_codes.clone().into_primitive();
                let m_scales = s.moment_1_scales.clone().into_primitive().tensor();
                let v_codes = s.moment_2_codes.clone().into_primitive();
                let v_scales = s.moment_2_scales.clone().into_primitive().tensor();
                (
                    m_codes.handle.clone(),
                    m_scales.handle.clone(),
                    v_codes.handle.clone(),
                    v_scales.handle.clone(),
                    s.time + 1,
                )
            }
            None => {
                let codes_bytes = num_blocks * PACKED_PER_BLOCK * core::mem::size_of::<u32>();
                let scales_bytes = num_blocks * core::mem::size_of::<f32>();

                // Zero-fill codes (decodes to zero), one-fill scales.
                let zero_codes_vec = vec![0u32; num_blocks * PACKED_PER_BLOCK];
                let one_scales_vec = vec![1.0f32; num_blocks];

                let m_codes = client.create(cubecl::bytes::Bytes::from_bytes_vec(
                    u32::as_bytes(&zero_codes_vec).to_vec(),
                ));
                let m_scales = client.create(cubecl::bytes::Bytes::from_bytes_vec(
                    f32::as_bytes(&one_scales_vec).to_vec(),
                ));
                let v_codes = client.create(cubecl::bytes::Bytes::from_bytes_vec(
                    u32::as_bytes(&zero_codes_vec).to_vec(),
                ));
                let v_scales = client.create(cubecl::bytes::Bytes::from_bytes_vec(
                    f32::as_bytes(&one_scales_vec).to_vec(),
                ));
                (m_codes, m_scales, v_codes, v_scales, 1)
            }
        };

        // Allocate output handles.
        let delta_handle = client.empty(padded_len * core::mem::size_of::<f32>());
        let m_codes_out_handle =
            client.empty(num_blocks * PACKED_PER_BLOCK * core::mem::size_of::<u32>());
        let m_scales_out_handle = client.empty(num_blocks * core::mem::size_of::<f32>());
        let v_codes_out_handle =
            client.empty(num_blocks * PACKED_PER_BLOCK * core::mem::size_of::<u32>());
        let v_scales_out_handle = client.empty(num_blocks * core::mem::size_of::<f32>());

        // Bias correction.
        let t = time as i32;
        let correction1 = 1.0f32 - self.beta_1.powi(t);
        let correction2_sqrt = (1.0f32 - self.beta_2.powi(t)).sqrt();
        let step_size = correction2_sqrt / correction1;
        let epsilon_eff = self.epsilon * correction2_sqrt;

        // Clone for both the launch (which consumes by-value) and any
        // subsequent need to wrap as a CubeTensor.
        let delta_for_launch = delta_handle.clone();
        let m_codes_for_launch = m_codes_out_handle.clone();
        let m_scales_for_launch = m_scales_out_handle.clone();
        let v_codes_for_launch = v_codes_out_handle.clone();
        let v_scales_for_launch = v_scales_out_handle.clone();

        let cube_count = cubecl::CubeCount::Static(num_blocks as u32, 1, 1);
        let cube_dim = cubecl::CubeDim::new_1d(BLOCK_SIZE as u32);

        unsafe {
            fused_adamw8bit_step_kernel::launch_unchecked::<R>(
                &client,
                cube_count,
                cube_dim,
                cubecl::prelude::ArrayArg::from_raw_parts(grad_handle, padded_len),
                cubecl::prelude::ArrayArg::from_raw_parts(
                    m_codes_handle,
                    num_blocks * PACKED_PER_BLOCK,
                ),
                cubecl::prelude::ArrayArg::from_raw_parts(m_scales_handle, num_blocks),
                cubecl::prelude::ArrayArg::from_raw_parts(
                    v_codes_handle,
                    num_blocks * PACKED_PER_BLOCK,
                ),
                cubecl::prelude::ArrayArg::from_raw_parts(v_scales_handle, num_blocks),
                cubecl::prelude::ArrayArg::from_raw_parts(delta_for_launch, padded_len),
                cubecl::prelude::ArrayArg::from_raw_parts(
                    m_codes_for_launch,
                    num_blocks * PACKED_PER_BLOCK,
                ),
                cubecl::prelude::ArrayArg::from_raw_parts(m_scales_for_launch, num_blocks),
                cubecl::prelude::ArrayArg::from_raw_parts(
                    v_codes_for_launch,
                    num_blocks * PACKED_PER_BLOCK,
                ),
                cubecl::prelude::ArrayArg::from_raw_parts(v_scales_for_launch, num_blocks),
                self.beta_1,
                self.beta_2,
                epsilon_eff,
                step_size,
            );
        }

        // Wrap the delta handle as a 1D padded CubeTensor, convert to Burn
        // Tensor, slice off padding, reshape to original.
        let delta_tensor_padded =
            wrap_handle_as_tensor_1d::<R>(&client, &device, delta_handle, padded_len);
        let delta = delta_tensor_padded
            .slice([0..total_elements])
            .reshape(Shape::from(original_shape));

        // Apply weight decay and lr (caller-side tensor ops). This matches
        // the existing AdamW8Bit's step() body.
        let decay_rate = lr * (self.weight_decay as f64);
        let decayed_tensor = if decay_rate == 0.0 {
            tensor
        } else {
            tensor.mul_scalar(1.0 - decay_rate)
        };
        let tensor_updated = decayed_tensor - delta.mul_scalar(lr);

        // Build the new state. Codes stay int-typed for Record save/load.
        let m_codes_out_tensor = wrap_handle_as_tensor_1d_int::<R>(
            &client,
            &device,
            m_codes_out_handle,
            num_blocks * PACKED_PER_BLOCK,
        );
        let m_scales_out_tensor =
            wrap_handle_as_tensor_1d::<R>(&client, &device, m_scales_out_handle, num_blocks);
        let v_codes_out_tensor = wrap_handle_as_tensor_1d_int::<R>(
            &client,
            &device,
            v_codes_out_handle,
            num_blocks * PACKED_PER_BLOCK,
        );
        let v_scales_out_tensor =
            wrap_handle_as_tensor_1d::<R>(&client, &device, v_scales_out_handle, num_blocks);

        let new_state = AdamW8BitFusedState {
            time,
            original_shape,
            moment_1_codes: m_codes_out_tensor,
            moment_1_scales: m_scales_out_tensor,
            moment_2_codes: v_codes_out_tensor,
            moment_2_scales: v_scales_out_tensor,
        };

        (tensor_updated, Some(new_state))
    }

    fn to_device<const D: usize>(
        mut state: Self::State<D>,
        device: &Device<CubeBackend<R>>,
    ) -> Self::State<D> {
        state.moment_1_codes = state.moment_1_codes.to_device(device);
        state.moment_1_scales = state.moment_1_scales.to_device(device);
        state.moment_2_codes = state.moment_2_codes.to_device(device);
        state.moment_2_scales = state.moment_2_scales.to_device(device);
        state
    }
}

impl AdamWConfig8BitFused {
    pub fn init<R, M>(&self) -> OptimizerAdaptor<AdamW8BitFused, M, B>
    where
        R: CubeRuntime,
        B: AutodiffBackend<InnerBackend = CubeBackend<R>>,
        M: AutodiffModule<B>,
    {
        let optim = AdamW8BitFused {
            beta_1: self.beta_1,
            beta_2: self.beta_2,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
        };
        let mut optim = OptimizerAdaptor::from(optim);
        if let Some(config) = &self.grad_clipping {
            optim = optim.with_grad_clipping(config.init());
        }
        optim
    }
}

// ---------------------------------------------------------------------------
// Helpers for going between raw Handles and Burn Tensors.
//
// These exist because the kernel operates on raw cubecl Handles, but the
// optimizer's input/output type is `Tensor<B, D>`. The wrap_handle_as_tensor_*
// helpers construct a CubeTensor from a fresh handle and known shape, then
// promote it to a Burn Tensor via TensorPrimitive::Float.
// ---------------------------------------------------------------------------

/// Pad an fp32 CubeTensor to a multiple of BLOCK_SIZE by reading to host,
/// padding with zeros, uploading. Returns the padded handle.
///
/// This is the slow path. A real optimization would do the padding on-device
/// or store state already padded.
fn pad_to_block_size<R: CubeRuntime>(
    client: &cubecl::ComputeClient<R>,
    grad: &CubeTensor<R>,
    padded_len: usize,
) -> Handle {
    let bytes = client.read_one_unchecked(grad.handle.clone());
    let raw: &[u8] = &bytes;
    let mut padded: Vec<f32> = f32::from_bytes(raw).to_vec();
    padded.resize(padded_len, 0.0f32);
    client.create(cubecl::bytes::Bytes::from_bytes_vec(
        f32::as_bytes(&padded).to_vec(),
    ))
}

/// Wrap a handle holding `n` fp32 elements into a Burn 1D float Tensor.
fn wrap_handle_as_tensor_1d<R: CubeRuntime>(
    client: &cubecl::ComputeClient<R>,
    device: &R::Device,
    handle: Handle,
    n: usize,
) -> Tensor<CubeBackend<R>, 1> {
    let cube_tensor = CubeTensor::<R>::new(
        client.clone(),
        handle,
        burn_cubecl::tensor::Metadata::new(Shape::from([n]), vec![1]),
        device.clone(),
        burn_cubecl::tensor::DType::F32,
        None,
    );
    Tensor::from_primitive(TensorPrimitive::Float(cube_tensor))
}

/// Wrap a handle holding `n` u32 elements into a Burn 1D int Tensor.
/// The codes tensor is logically u8 packed 4-per-word, but we store as Int
/// so it round-trips through Record cleanly.
fn wrap_handle_as_tensor_1d_int<R: CubeRuntime>(
    client: &cubecl::ComputeClient<R>,
    device: &R::Device,
    handle: Handle,
    n: usize,
) -> Tensor<CubeBackend<R>, 1, Int> {
    let cube_tensor = CubeTensor::<R>::new(
        client.clone(),
        handle,
        burn_cubecl::tensor::Metadata::new(Shape::from([n]), vec![1]),
        device.clone(),
        burn_cubecl::tensor::DType::U32,
        None,
    );
    Tensor::from_primitive(cube_tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Tests will go here once Stage 2 compiles. The plan is to mirror the
    // existing adamw_8bit test suite with TestAutodiffBackend = CubeBackend
    // (CUDA) and assert equivalent param updates.
}
