use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{tensor::CubeTensor, CubeRuntime, FloatElement};

#[cube(launch_unchecked)]
fn interpolate_nearest_kernel<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let batch = ABSOLUTE_POS / output.stride(0) % output.shape(0);
    let channel = ABSOLUTE_POS / output.stride(1) % output.shape(1);
    let y = ABSOLUTE_POS / output.stride(2) % output.shape(2);
    let x = ABSOLUTE_POS / output.stride(3) % output.shape(3);

    let factor = F::cast_from(y);
    let numerator = F::cast_from(input.shape(2));
    let denominator = F::cast_from(output.shape(2));
    let y: F = Floor::floor(factor * numerator / denominator);

    let factor = F::cast_from(x);
    let numerator = F::cast_from(input.shape(3));
    let denominator = F::cast_from(output.shape(3));
    let x: F = Floor::floor(factor * numerator / denominator);

    let index = batch * input.stride(0)
        + channel * input.stride(1)
        + u32::cast_from(y) * input.stride(2)
        + u32::cast_from(x) * input.stride(3);

    output[ABSOLUTE_POS] = input[index];
}

pub(crate) fn interpolate_nearest_launch<R: CubeRuntime, E: FloatElement>(
    input: CubeTensor<R>,
    output: CubeTensor<R>,
) -> CubeTensor<R> {
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(output.shape.num_elements(), cube_dim);

    unsafe {
        interpolate_nearest_kernel::launch_unchecked::<E, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg::<E>(1),
            output.as_tensor_arg::<E>(1),
        )
    };

    output
}
