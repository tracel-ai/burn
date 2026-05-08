mod blockwise;
mod signed;
mod transform;
mod unsigned;
mod weight_decay;

use burn_core as burn;

use cubecl::prelude::*;

use crate::launch::PLANE_SIZE;

#[cube(launch_unchecked)]
pub fn adamw_8bit_step_kernel(
    // ---- Inputs (read) ----
    theta: &Array<f32>,
    grad: &Array<f32>,

    moment_1_codes: &Array<u32>,
    moment_1_scales: &Array<f32>,
    moment_2_codes: &Array<u32>,
    moment_2_scales: &Array<f32>,
    max_moment_2_codes: &Array<u32>,
    max_moment_2_scales: &Array<f32>,

    // ---- Outputs (write) ----
    theta_new: &mut Array<f32>,

    moment_1_codes_new: &mut Array<u32>,
    moment_1_scales_new: &mut Array<f32>,
    moment_2_codes_new: &mut Array<u32>,
    moment_2_scales_new: &mut Array<f32>,
    max_moment_2_codes_new: &mut Array<u32>,
    max_moment_2_scales_new: &mut Array<f32>,

    // ---- Scratch (per-step intermediates) ----
    update_delta: &mut Array<f32>,
    m1_dequantized: &mut Array<f32>,

    // ---- Runtime scalars ----
    beta_1: f32,
    beta_2: f32,
    factor_1: f32,
    factor_2: f32,
    correction1: f32,
    correction2_sqrt: f32,
    epsilon: f32,
    lr: f32,
    decay_rate: f32,

    // ---- Compile-time specialization ----
    #[comptime] block_size: u32,
    #[comptime] amsgrad: bool,
    #[comptime] is_first_step: bool,
    #[comptime] cautious_weight_decay: bool,
) {
    let block = CUBE_POS_X;
    let unit = UNIT_POS_X;
    let i = block * block_size + unit;

    // --- Phase 1: Moment update + delta computation + requantization ---
    transform::transform(
        block,
        grad,
        moment_1_codes,
        moment_1_scales,
        moment_2_codes,
        moment_2_scales,
        max_moment_2_codes,
        max_moment_2_scales,
        moment_1_codes_new,
        moment_1_scales_new,
        moment_2_codes_new,
        moment_2_scales_new,
        max_moment_2_codes_new,
        max_moment_2_scales_new,
        update_delta,
        m1_dequantized,
        beta_1,
        beta_2,
        factor_1,
        factor_2,
        correction1,
        correction2_sqrt,
        epsilon,
        block_size,
        amsgrad,
        is_first_step,
    );

    // --- Phase 2: Weight decay + parameter update ---
    // weight_decay::weight_decay(
    //     i,
    //     theta,
    //     update_delta,
    //     m1_dequantized,
    //     theta_new,
    //     lr,
    //     decay_rate,
    //     cautious_weight_decay,
    // );
    let elements_per_thread = comptime!(block_size / PLANE_SIZE);

    #[unroll]
    for iter in 0..elements_per_thread {
        let element = unit + iter * PLANE_SIZE;
        let i = block * block_size + element;
        weight_decay::weight_decay(
            i,
            theta,
            update_delta,
            m1_dequantized,
            theta_new,
            lr,
            decay_rate,
            cautious_weight_decay,
        );
    }
}
