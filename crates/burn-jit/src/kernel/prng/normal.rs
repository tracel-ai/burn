use cubecl::prelude::*;
use std::f32::consts::PI;

use burn_tensor::Shape;

use crate::{
    kernel::prng::{cast_uint_to_float, lcg_step, taus_step_0, taus_step_1, taus_step_2},
    tensor::JitTensor,
    JitElement, JitRuntime,
};

use super::{random, Prng, PrngRuntime};

pub(crate) struct Normal<E> {
    mean: E,
    std: E,
}

#[cube]
impl<E: JitElement> PrngRuntime<E> for Normal<E> {
    fn inner_loop(
        args: Sequence<E>,
        write_index_base: u32,
        n_invocations: u32,
        #[comptime] n_values_per_thread: u32,
        state_0: &mut u32,
        state_1: &mut u32,
        state_2: &mut u32,
        state_3: &mut u32,
        output: &mut Tensor<E>,
    ) {
        let mean = f32::cast_from(*args.index(0));
        let std = f32::cast_from(*args.index(0));

        let should_unroll = n_values_per_thread <= 16;

        #[unroll(should_unroll)]
        for i in 0..n_values_per_thread / 2 {
            // First random uniform integer
            *state_0 = taus_step_0(*state_0);
            *state_1 = taus_step_1(*state_1);
            *state_2 = taus_step_2(*state_2);
            *state_3 = lcg_step(*state_3);

            let int_random = *state_0 ^ *state_1 ^ *state_2 ^ *state_3;
            let unit_0 = cast_uint_to_float(int_random);

            // Second random uniform integer
            *state_0 = taus_step_0(*state_0);
            *state_1 = taus_step_1(*state_1);
            *state_2 = taus_step_2(*state_2);
            *state_3 = lcg_step(*state_3);

            let int_random = *state_0 ^ *state_1 ^ *state_2 ^ *state_3;
            let unit_1 = cast_uint_to_float(int_random);

            // Box-Muller transform
            let coeff = Log::log(unit_0) * -2.0;
            let coeff = Sqrt::sqrt(coeff) * std;
            let trigo_arg = 2.0 * PI * unit_1;

            let normal_0 = f32::cos(trigo_arg) * coeff + mean;
            let normal_1 = f32::sin(trigo_arg) * coeff + mean;

            // Write to output
            let iteration_offset = 2 * i * n_invocations;
            let write_index_0 = write_index_base + iteration_offset;
            let write_index_1 = write_index_0 + n_invocations;

            output[write_index_0] = E::cast_from(normal_0);
            output[write_index_1] = E::cast_from(normal_1);
        }
    }
}

impl<E: JitElement> Prng<E> for Normal<E> {
    fn args(self) -> Vec<E> {
        vec![self.mean, self.std]
    }
}

/// Pseudo-random generator with uniform distribution
pub fn random_normal<R: JitRuntime, E: JitElement>(
    shape: Shape,
    device: &R::Device,
    mean: E,
    std: E,
) -> JitTensor<R, E> {
    random(shape, device, Normal { mean, std })
}
