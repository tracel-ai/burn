use burn_tensor::Shape;
use cubecl::prelude::*;

use crate::{
    kernel::prng::{cast_uint_to_float, lcg_step, taus_step_0, taus_step_1, taus_step_2},
    tensor::JitTensor,
    JitElement, JitRuntime,
};

use super::{random, Prng, PrngRuntime};

pub(crate) struct Uniform<E> {
    lower_bound: E,
    upper_bound: E,
}

#[cube]
impl<E: JitElement> PrngRuntime<E> for Uniform<E> {
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
        let lower_bound = *args.index(0);
        let upper_bound = *args.index(1);

        let should_unroll = n_values_per_thread <= 8;
        let scale = upper_bound - lower_bound;

        #[unroll(should_unroll)]
        for i in 0..n_values_per_thread {
            *state_0 = taus_step_0(*state_0);
            *state_1 = taus_step_1(*state_1);
            *state_2 = taus_step_2(*state_2);
            *state_3 = lcg_step(*state_3);

            let int_random = *state_0 ^ *state_1 ^ *state_2 ^ *state_3;
            let f32_random = cast_uint_to_float(int_random);
            let random = E::cast_from(f32_random);

            let uniform = random * scale + lower_bound;

            let write_index = i * n_invocations + write_index_base;
            output[write_index] = uniform;
        }
    }
}

impl<E: JitElement> Prng<E> for Uniform<E> {
    fn args(self) -> Vec<E> {
        vec![self.lower_bound, self.upper_bound]
    }
}

/// Pseudo-random generator with uniform distribution
pub fn random_uniform<R: JitRuntime, E: JitElement>(
    shape: Shape,
    device: &R::Device,
    lower_bound: E,
    upper_bound: E,
) -> JitTensor<R, E> {
    random(
        shape,
        device,
        Uniform {
            lower_bound,
            upper_bound,
        },
    )
}
/// Pseudo-random generator for uniform distribution, based on
/// another tensor.
pub fn random_like_uniform<R: JitRuntime, E: JitElement>(
    tensor: &JitTensor<R, E>,
    lower_bound: E,
    upper_bound: E,
) -> JitTensor<R, E> {
    random_uniform(
        tensor.shape.clone(),
        &tensor.device,
        lower_bound,
        upper_bound,
    )
}
