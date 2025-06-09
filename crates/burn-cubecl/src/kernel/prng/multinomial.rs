use burn_tensor::Shape;
use cubecl::prelude::*;

use crate::{
    kernel::prng::{cast_uint_to_float, lcg_step, taus_step_0, taus_step_1, taus_step_2},
    ops::numeric::empty_device,
    tensor::CubeTensor,
    CubeElement, CubeRuntime,
};

use super::{get_seeds, prng_cube_count, N_VALUES_PER_THREAD};

/// Pseudo-random generator
fn random<R: CubeRuntime, E: CubeElement>(
    shape: Shape,
    device: &R::Device,
    args: TensorArg<'_, R>,
) -> CubeTensor<R> {
    let client = R::client(device);
    let output = empty_device::<R, E>(client.clone(), device.clone(), shape);
    let seeds = get_seeds();

    let cube_dim = CubeDim::default();
    let cube_count = prng_cube_count(output.shape.num_elements(), cube_dim, N_VALUES_PER_THREAD);

    prng_kernel::launch::<E, R>(
        &client,
        cube_count,
        cube_dim,
        output.as_tensor_arg::<E>(1),
        ScalarArg::new(seeds[0]),
        ScalarArg::new(seeds[1]),
        ScalarArg::new(seeds[2]),
        ScalarArg::new(seeds[3]),
        args,
        N_VALUES_PER_THREAD as u32,
    );

    output
}
#[cube(launch)]
fn prng_kernel<E: CubeElement>(
    output: &mut Tensor<E>,
    seed_0: u32,
    seed_1: u32,
    seed_2: u32,
    seed_3: u32,
    args: Tensor<E>,
    #[comptime] n_values_per_thread: u32,
) {
    let cube_offset = CUBE_POS * CUBE_DIM;

    let write_index_base = cube_offset * n_values_per_thread + UNIT_POS;

    #[allow(arithmetic_overflow)]
    let thread_seed = 1000000007u32 * ABSOLUTE_POS;

    let mut state_0 = thread_seed + seed_0;
    let mut state_1 = thread_seed + seed_1;
    let mut state_2 = thread_seed + seed_2;
    let mut state_3 = thread_seed + seed_3;
    let n_invocations = CUBE_DIM;

    // Creation of n_values_per_thread values, specific to the distribution
    let prob = 23.0; // TODO
    let should_unroll = n_values_per_thread <= 8;

    #[unroll(should_unroll)]
    for i in 0..n_values_per_thread {
        state_0 = taus_step_0(state_0);
        state_1 = taus_step_1(state_1);
        state_2 = taus_step_2(state_2);
        state_3 = lcg_step(state_3);

        let int_random = state_0 ^ state_1 ^ state_2 ^ state_3;
        let float_random = cast_uint_to_float(int_random);
        let write_index = i * n_invocations + write_index_base;

        output[write_index] = E::cast_from(float_random < prob);
    }
}

#[derive(CubeLaunch)]
struct Multinomial<E: Numeric> {
    probabilities: Tensor<E>,
}

/// Pseudo-random generator with uniform distribution
pub fn random_multinomial<R: CubeRuntime, E: CubeElement>(
    shape: Shape,
    device: &R::Device,
    props: CubeTensor<R>,
) -> CubeTensor<R> {
    random::<R, E>(shape, device, props.as_tensor_arg::<E>(1))
}

/// Pseudo-random generator for uniform distribution, based on
/// another tensor.
pub fn random_like_multinomial<R: CubeRuntime, E: CubeElement>(
    tensor: &CubeTensor<R>,
    props: CubeTensor<R>,
) -> CubeTensor<R> {
    random_multinomial::<R, E>(tensor.shape.clone(), &tensor.device, props)
}
