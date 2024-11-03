use burn_tensor::Shape;
use cubecl::prelude::*;

use crate::{
    kernel::prng::{cast_uint_to_float, lcg_step, taus_step_0, taus_step_1, taus_step_2},
    tensor::JitTensor,
    JitElement, JitRuntime,
};

use super::{random, Prng, PrngRuntime};

pub(crate) struct Bernoulli<E> {
    probability: E,
}

#[cube]
impl<E: JitElement> PrngRuntime<E> for Bernoulli<E> {
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
        let prob = f32::cast_from(*args.index(0));
        let should_unroll = n_values_per_thread <= 8;

        #[unroll(should_unroll)]
        for i in 0..n_values_per_thread {
            *state_0 = taus_step_0(*state_0);
            *state_1 = taus_step_1(*state_1);
            *state_2 = taus_step_2(*state_2);
            *state_3 = lcg_step(*state_3);

            let int_random = *state_0 ^ *state_1 ^ *state_2 ^ *state_3;
            let float_random = cast_uint_to_float(int_random);

            let write_index = i * n_invocations + write_index_base;
            output[write_index] = E::cast_from(float_random < prob);
        }
    }
}

impl<E: JitElement> Prng<E> for Bernoulli<E> {
    fn args(self) -> Vec<E> {
        vec![self.probability]
    }
}

/// Pseudo-random generator with bernoulli distribution
pub fn random_bernoulli<R: JitRuntime, E: JitElement>(
    shape: Shape,
    device: &R::Device,
    probability: E,
) -> JitTensor<R, E> {
    random(shape, device, Bernoulli { probability })
}
