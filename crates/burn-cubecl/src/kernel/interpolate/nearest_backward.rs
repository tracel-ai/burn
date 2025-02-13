use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{tensor::CubeTensor, CubeRuntime, FloatElement};

#[cube(launch_unchecked)]
fn interpolate_nearest_backward_kernel<F: Float>(grad: &Tensor<F>, output: &mut Tensor<F>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let out_h = output.shape(2);
    let out_w = output.shape(3);
    let grad_h = grad.shape(2);
    let grad_w = grad.shape(3);

    let batch = ABSOLUTE_POS / output.stride(0) % output.shape(0);
    let channel = ABSOLUTE_POS / output.stride(1) % output.shape(1);
    let oh = ABSOLUTE_POS / output.stride(2) % out_h;
    let ow = ABSOLUTE_POS / output.stride(3) % out_w;

    let gh_start = start_index::<F>(oh, grad_h, out_h);
    let gh_end = end_index::<F>(oh, grad_h, out_h);
    let gw_start = start_index::<F>(ow, grad_w, out_w);
    let gw_end = end_index::<F>(ow, grad_w, out_w);

    let index_grad_base = batch * grad.stride(0) + channel * grad.stride(1);

    let mut sum = F::new(0.0);

    for gh in gh_start..gh_end {
        for gw in gw_start..gw_end {
            let index_grad = index_grad_base + gh * grad.stride(2) + gw * grad.stride(3);

            sum += grad[index_grad];
        }
    }

    output[ABSOLUTE_POS] = sum;
}

#[cube]
fn start_index<F: Float>(input_index: u32, output_size: u32, input_size: u32) -> u32 {
    let numerator = F::cast_from(input_index * output_size);
    let div: F = Ceil::ceil(numerator / F::cast_from(input_size));

    u32::cast_from(div)
}

#[cube]
fn end_index<F: Float>(input_index: u32, output_size: u32, input_size: u32) -> u32 {
    let numerator = F::cast_from((input_index + 1) * output_size);
    let div: F = Ceil::ceil(numerator / F::cast_from(input_size));
    let index = u32::cast_from(div);

    Min::min(output_size, index)
}

pub(crate) fn interpolate_nearest_backward_launch<R: CubeRuntime, E: FloatElement>(
    out_grad: CubeTensor<R>,
    output: CubeTensor<R>,
) -> CubeTensor<R> {
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(output.shape.num_elements(), cube_dim);

    unsafe {
        interpolate_nearest_backward_kernel::launch_unchecked::<E, R>(
            &out_grad.client,
            cube_count,
            cube_dim,
            out_grad.as_tensor_arg::<E>(1),
            output.as_tensor_arg::<E>(1),
        )
    };

    output
}
