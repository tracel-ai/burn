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
    // All in-place now
    theta: &mut Array<f32>,
    grad: &Array<f32>,
    moment_1_codes: &mut Array<u32>,
    moment_1_scales: &mut Array<f32>,
    moment_2_codes: &mut Array<u32>,
    moment_2_scales: &mut Array<f32>,
    max_moment_2_codes: &mut Array<u32>,
    max_moment_2_scales: &mut Array<f32>,
    // Runtime scalars
    beta_1: f32,
    beta_2: f32,
    factor_1: f32,
    factor_2: f32,
    correction1: f32,
    correction2_sqrt: f32,
    epsilon: f32,
    lr: f32,
    decay_rate: f32,
    #[comptime] block_size: u32,
    #[comptime] amsgrad: bool,
    #[comptime] is_first_step: bool,
    #[comptime] cautious_weight_decay: bool,
) {
    let block = CUBE_POS_X;
    let unit = UNIT_POS_X;
    let elements_per_thread = comptime!(block_size / PLANE_SIZE);

    // Per-lane registers — these replace update_delta and m1_dequantized globals.
    // cubecl supports Array<f32, N> as a register-resident local array when N
    // is comptime. If your cubecl version doesn't, use a sequence of scalars
    // unrolled, or shared memory of size block_size.
    let mut local_delta = Array::<f32>::new(elements_per_thread as usize);
    let mut local_m1 = Array::<f32>::new(elements_per_thread as usize);

    // Phase 1: transform writes into locals instead of globals.
    transform::transform(
        block,
        grad,
        moment_1_codes,
        moment_1_scales,
        moment_2_codes,
        moment_2_scales,
        max_moment_2_codes,
        max_moment_2_scales,
        &mut local_delta,
        &mut local_m1,
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

    // Phase 2: weight decay + in-place theta update, reading deltas from registers.
    #[unroll]
    for iter in 0..elements_per_thread {
        let element = unit + iter * PLANE_SIZE;
        let i = block * block_size + element;
        weight_decay::weight_decay(
            i,
            iter,
            theta,
            &local_delta,
            &local_m1,
            lr,
            decay_rate,
            cautious_weight_decay,
        );
    }
}
