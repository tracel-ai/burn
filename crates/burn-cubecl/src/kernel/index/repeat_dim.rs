use crate::{CubeRuntime, element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

#[cube(launch_unchecked)]
fn repeat_dim_kernel<E: CubePrimitive>(input: &Tensor<E>, output: &mut Tensor<E>, dim: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let mut offset_input = 0;

    for i in 0..input.rank() {
        let shape = select(i != dim, output.shape(i), input.shape(i));

        let offset_local = ABSOLUTE_POS / output.stride(i) % shape * input.stride(i);
        offset_input += offset_local;
    }

    output[ABSOLUTE_POS] = input[offset_input];
}

pub(crate) fn repeat_dim<R: CubeRuntime, E: CubeElement>(
    mut input: CubeTensor<R>,
    dim: usize,
    times: usize,
) -> CubeTensor<R> {
    if input.shape[dim] == 1 {
        input.strides[dim] = 0;
        input.shape = input.shape.repeat(dim, times);
        return input;
    }

    let shape = input.shape.clone().repeat(dim, times);

    // Create output handle
    let output = empty_device::<R, E>(input.client.clone(), input.device.clone(), shape);

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(output.shape.num_elements(), cube_dim);

    unsafe {
        repeat_dim_kernel::launch_unchecked::<E, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg::<E>(1),
            output.as_tensor_arg::<E>(1),
            ScalarArg::new(dim as u32),
        )
    };

    output
}
