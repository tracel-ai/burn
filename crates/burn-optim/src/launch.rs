use cubecl::prelude::*;

use crate::kernel::adamw_8bit_step_kernel;

pub const PACKING_AMOUNT: u32 = 4;
/// Don't mess with this value. I know it doesn't work on AMD.
pub const PLANE_SIZE: u32 = 32;
pub const PACK_SHIFT: u32 = 2_u32.pow(8);

// /// Result of one fused step: the updated parameter and the new state buffers.
// pub struct AdamWStepOutput<R: Runtime> {
//     pub theta_new: cubecl::server::Handle,
//     pub moment_1_codes: cubecl::server::Handle,
//     pub moment_1_scales: cubecl::server::Handle,
//     pub moment_2_codes: cubecl::server::Handle,
//     pub moment_2_scales: cubecl::server::Handle,
//     pub max_moment_2_codes: Option<cubecl::server::Handle>,
//     pub max_moment_2_scales: Option<cubecl::server::Handle>,
//     pub _phantom: core::marker::PhantomData<R>,
// }

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

pub struct AdamWStepOutput;

pub fn launch_adamw_8bit_step<R: Runtime>(
    client: &ComputeClient<R>,
    inputs: AdamWStepInputs<R>,
    params: AdamWStepParams,
) -> AdamWStepOutput {
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

    let factor_1 = 1.0 - params.beta_1;
    let factor_2 = 1.0 - params.beta_2;
    let time = params.time as i32;
    let correction1 = 1.0 - params.beta_1.powi(time);
    let correction2_sqrt = (1.0 - params.beta_2.powi(time)).sqrt();
    let decay_rate = params.lr * params.weight_decay;

    // Caller guarantees state buffers exist (they're allocated once on the
    // optimizer's first step and reused forever). is_first_step tells the
    // kernel to ignore their contents and initialize from zero.
    let is_first_step = params.time == 1;

    let m1_codes = inputs.moment_1_codes.expect("state must be preallocated");
    let m1_scales = inputs.moment_1_scales.expect("state must be preallocated");
    let m2_codes = inputs.moment_2_codes.expect("state must be preallocated");
    let m2_scales = inputs.moment_2_scales.expect("state must be preallocated");

    let (max_v_codes, max_v_scales, max_v_codes_size, max_v_scales_size) = if params.amsgrad {
        (
            inputs.max_moment_2_codes.expect("amsgrad state required"),
            inputs.max_moment_2_scales.expect("amsgrad state required"),
            packed_count,
            num_blocks_usize,
        )
    } else {
        // Tiny dummy buffers — kernel never touches them under comptime branch.
        (
            client.empty(core::mem::size_of::<u32>()),
            client.empty(core::mem::size_of::<f32>()),
            1,
            1,
        )
    };

    unsafe {
        adamw_8bit_step_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(num_blocks, 1, 1),
            CubeDim::new(client, PLANE_SIZE as usize),
            // All in-place: same buffer as input and "output"
            ArrayArg::from_raw_parts(inputs.theta, n),
            ArrayArg::from_raw_parts(inputs.grad, n),
            ArrayArg::from_raw_parts(m1_codes, packed_count),
            ArrayArg::from_raw_parts(m1_scales, num_blocks_usize),
            ArrayArg::from_raw_parts(m2_codes, packed_count),
            ArrayArg::from_raw_parts(m2_scales, num_blocks_usize),
            ArrayArg::from_raw_parts(max_v_codes, max_v_codes_size),
            ArrayArg::from_raw_parts(max_v_scales, max_v_scales_size),
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

    AdamWStepOutput
}
