use cubecl::prelude::*;

use crate::kernel::adamw_8bit_step_kernel;

pub const PACKING_AMOUNT: u32 = 2;
/// Don't mess with this value. I know it doesn't work on AMD.
pub const PLANE_SIZE: u32 = 32;
pub const PACK_SHIFT: u32 = 256;

/// Result of one fused step: the updated parameter and the new state buffers.
pub struct AdamWStepOutput<R: Runtime> {
    pub theta_new: cubecl::server::Handle,
    pub moment_1_codes: cubecl::server::Handle,
    pub moment_1_scales: cubecl::server::Handle,
    pub moment_2_codes: cubecl::server::Handle,
    pub moment_2_scales: cubecl::server::Handle,
    pub max_moment_2_codes: Option<cubecl::server::Handle>,
    pub max_moment_2_scales: Option<cubecl::server::Handle>,
    pub _phantom: core::marker::PhantomData<R>,
}

/// Hyperparameters for one step.
pub struct AdamWStepParams {
    pub beta_1: f32,
    pub beta_2: f32,
    pub epsilon: f32,
    pub lr: f32,
    pub weight_decay: f32,
    pub time: u32,
    pub block_size: u32,
    pub amsgrad: bool,
    pub cautious_weight_decay: bool,
}

/// Per-element layout. The caller knows how many elements there are; the
/// scales arrays are sized as numel / block_size, and the codes arrays
/// as numel / PACKING_AMOUNT.
pub struct AdamWStepInputs<R: Runtime> {
    pub theta: cubecl::server::Handle,
    pub grad: cubecl::server::Handle,
    pub moment_1_codes: Option<cubecl::server::Handle>,
    pub moment_1_scales: Option<cubecl::server::Handle>,
    pub moment_2_codes: Option<cubecl::server::Handle>,
    pub moment_2_scales: Option<cubecl::server::Handle>,
    pub max_moment_2_codes: Option<cubecl::server::Handle>,
    pub max_moment_2_scales: Option<cubecl::server::Handle>,
    pub numel: usize,
    pub _phantom: core::marker::PhantomData<R>,
}

/// Launch one fused AdamW step.
///
/// Returns handles to the new parameter and new state buffers. The caller
/// is responsible for wrapping these back into Burn tensors / state structs.
pub fn launch_adamw_8bit_step<R: Runtime>(
    client: &ComputeClient<R>,
    inputs: AdamWStepInputs<R>,
    params: AdamWStepParams,
) -> AdamWStepOutput<R> {
    let n = inputs.numel;
    assert_eq!(
        n % params.block_size as usize,
        0,
        "numel ({}) must be a multiple of block_size ({})",
        n,
        params.block_size,
    );

    let num_blocks = (n as u32) / params.block_size;
    let packed_count = n / PACKING_AMOUNT as usize;
    let num_blocks_usize = num_blocks as usize;

    // ---- Host-side scalar precomputation ----
    let factor_1 = 1.0 - params.beta_1;
    let factor_2 = 1.0 - params.beta_2;
    let time = params.time as i32;
    let correction1 = 1.0 - params.beta_1.powi(time);
    let correction2_sqrt = (1.0 - params.beta_2.powi(time)).sqrt();
    let decay_rate = params.lr * params.weight_decay;
    let is_first_step = inputs.moment_1_codes.is_none();

    // ---- Allocate output buffers ----
    let theta_new_handle = client.empty(n * core::mem::size_of::<f32>());
    let m1_codes_new = client.empty(packed_count * core::mem::size_of::<u32>());
    let m1_scales_new = client.empty(num_blocks_usize * core::mem::size_of::<f32>());
    let m2_codes_new = client.empty(packed_count * core::mem::size_of::<u32>());
    let m2_scales_new = client.empty(num_blocks_usize * core::mem::size_of::<f32>());

    // For !amsgrad, allocate dummy 1-element buffers; the comptime branch
    // ensures the kernel never reads or writes them meaningfully.
    let max_v_codes_new = if params.amsgrad {
        client.empty(packed_count * core::mem::size_of::<u32>())
    } else {
        client.empty(core::mem::size_of::<u32>())
    };
    let max_v_scales_new = if params.amsgrad {
        client.empty(num_blocks_usize * core::mem::size_of::<f32>())
    } else {
        client.empty(core::mem::size_of::<f32>())
    };

    // ---- Per-step scratch buffers ----
    // delta and m1_dequantized are intermediate values used between the
    // transform and weight_decay phases. They're sized to the parameter.
    let delta_handle = client.empty(n * core::mem::size_of::<f32>());
    let m1_dequant_handle = client.empty(n * core::mem::size_of::<f32>());

    // ---- Resolve input handles ----
    // On the first step, moments are None — pass dummy handles (the kernel
    // skips the dequantize path under is_first_step=true).
    let dummy_codes = client.empty(core::mem::size_of::<u32>());
    let dummy_scales = client.empty(core::mem::size_of::<f32>());

    let m1_codes_in = inputs.moment_1_codes.unwrap_or_else(|| dummy_codes.clone());
    let m1_scales_in = inputs
        .moment_1_scales
        .unwrap_or_else(|| dummy_scales.clone());
    let m2_codes_in = inputs.moment_2_codes.unwrap_or_else(|| dummy_codes.clone());
    let m2_scales_in = inputs
        .moment_2_scales
        .unwrap_or_else(|| dummy_scales.clone());
    let max_v_codes_in = inputs
        .max_moment_2_codes
        .unwrap_or_else(|| dummy_codes.clone());
    let max_v_scales_in = inputs
        .max_moment_2_scales
        .unwrap_or_else(|| dummy_scales.clone());

    // Sizes for ArrayArg::from_raw_parts. For dummy handles used in
    // unreachable branches, just use 1 — the kernel won't touch them.
    let m1_codes_in_size = if is_first_step { 1 } else { packed_count };
    let m1_scales_in_size = if is_first_step { 1 } else { num_blocks_usize };
    let m2_codes_in_size = if is_first_step { 1 } else { packed_count };
    let m2_scales_in_size = if is_first_step { 1 } else { num_blocks_usize };
    let max_v_codes_in_size = if params.amsgrad && !is_first_step {
        packed_count
    } else {
        1
    };
    let max_v_scales_in_size = if params.amsgrad && !is_first_step {
        num_blocks_usize
    } else {
        1
    };
    let max_v_codes_out_size = if params.amsgrad { packed_count } else { 1 };
    let max_v_scales_out_size = if params.amsgrad { num_blocks_usize } else { 1 };

    // ---- Launch ----
    unsafe {
        adamw_8bit_step_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(num_blocks, 1, 1),
            CubeDim::new(client, PLANE_SIZE as usize),
            ArrayArg::from_raw_parts(inputs.theta, n),
            ArrayArg::from_raw_parts(inputs.grad, n),
            ArrayArg::from_raw_parts(m1_codes_in, m1_codes_in_size),
            ArrayArg::from_raw_parts(m1_scales_in, m1_scales_in_size),
            ArrayArg::from_raw_parts(m2_codes_in, m2_codes_in_size),
            ArrayArg::from_raw_parts(m2_scales_in, m2_scales_in_size),
            ArrayArg::from_raw_parts(max_v_codes_in, max_v_codes_in_size),
            ArrayArg::from_raw_parts(max_v_scales_in, max_v_scales_in_size),
            ArrayArg::from_raw_parts(theta_new_handle.clone(), n),
            ArrayArg::from_raw_parts(m1_codes_new.clone(), packed_count),
            ArrayArg::from_raw_parts(m1_scales_new.clone(), num_blocks_usize),
            ArrayArg::from_raw_parts(m2_codes_new.clone(), packed_count),
            ArrayArg::from_raw_parts(m2_scales_new.clone(), num_blocks_usize),
            ArrayArg::from_raw_parts(max_v_codes_new.clone(), max_v_codes_out_size),
            ArrayArg::from_raw_parts(max_v_scales_new.clone(), max_v_scales_out_size),
            ArrayArg::from_raw_parts(delta_handle, n),
            ArrayArg::from_raw_parts(m1_dequant_handle, n),
            params.beta_1,
            params.beta_2,
            factor_1,
            factor_2,
            correction1,
            correction2_sqrt,
            params.epsilon,
            params.lr,
            decay_rate,
            params.block_size,
            params.amsgrad,
            is_first_step,
            params.cautious_weight_decay,
        );
    }

    AdamWStepOutput {
        theta_new: theta_new_handle,
        moment_1_codes: m1_codes_new,
        moment_1_scales: m1_scales_new,
        moment_2_codes: m2_codes_new,
        moment_2_scales: m2_scales_new,
        max_moment_2_codes: if params.amsgrad {
            Some(max_v_codes_new)
        } else {
            None
        },
        max_moment_2_scales: if params.amsgrad {
            Some(max_v_scales_new)
        } else {
            None
        },
        _phantom: core::marker::PhantomData,
    }
}
