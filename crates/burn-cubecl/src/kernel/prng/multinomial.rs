use burn_tensor::Shape;
use cubecl::{linalg::tensor::TensorHandle, prelude::*};

use crate::{
    kernel::prng::{cast_uint_to_float, lcg_step, taus_step_0, taus_step_1, taus_step_2},
    tensor::CubeTensor,
    CubeElement, CubeRuntime,
};

use super::{random, PrngArgs, PrngRuntime};

#[derive(CubeLaunch)]
pub(crate) struct Multinomial<E: Numeric + CubeType> {
    props: Tensor<E>,
}

#[cube]
impl<E: CubeElement + CubeType> PrngRuntime<E> for Multinomial<E> {
    fn inner_loop(
        args: Multinomial<E>,
        write_index_base: u32,
        n_invocations: u32,
        #[comptime] n_values_per_thread: u32,
        state_0: &mut u32,
        state_1: &mut u32,
        state_2: &mut u32,
        state_3: &mut u32,
        output: &mut Tensor<E>,
    ) {
        let props = args.props;

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


/// Pseudo-random generator with uniform distribution
pub fn random_multinomial<R: CubeRuntime, E: CubeElement>(
    shape: Shape,
    device: &R::Device,
    props: Tensor<E>,
) -> CubeTensor<R> {
    random(shape, device, Multinomial { props })
}
/// Pseudo-random generator for uniform distribution, based on
/// another tensor.
pub fn random_like_multinomial<R: CubeRuntime, E: CubeElement>(
    tensor: &CubeTensor<R>,
    props: Tensor<E>,
) -> CubeTensor<R> {
    random_multinomial::<R, E>(tensor.shape.clone(), &tensor.device, props)
}
