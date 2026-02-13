use cubecl::std::{
    FastDivmod,
    tensor::layout::{linear::LinearLayout, *},
};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{
    CubeRuntime,
    kernel::utils::{address_type, linear_layout, shape_divmod},
    ops::max_line_size,
    tensor::CubeTensor,
};

#[cube(launch_unchecked, address_type = "dynamic")]
fn interpolate_nearest_backward_kernel<F: Float>(
    grad: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    shape_out: Sequence<FastDivmod<usize>>,
    out_layout: LinearLayout,
    #[define(F)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let line_size = grad.line_size();
    let out_idx = out_layout.to_source_pos(ABSOLUTE_POS);

    let out_h = output.shape(1);
    let out_w = output.shape(2);
    let grad_h = grad.shape(1);
    let grad_w = grad.shape(2);

    let (rem, c) = shape_out[3].div_mod(ABSOLUTE_POS * line_size);
    let (rem, out_x) = shape_out[2].div_mod(rem);
    let (b, out_y) = shape_out[1].div_mod(rem);

    let grad_y_start = start_index::<F>(out_y, grad_h, out_h);
    let grad_y_end = end_index::<F>(out_y, grad_h, out_h);
    let grad_x_start = start_index::<F>(out_x, grad_w, out_w);
    let grad_x_end = end_index::<F>(out_x, grad_w, out_w);

    let index_grad_base = b * grad.stride(0) + c * grad.stride(3);

    let mut sum = Line::empty(line_size).fill(F::new(0.0));

    for grad_y in grad_y_start..grad_y_end {
        for grad_x in grad_x_start..grad_x_end {
            let index_grad = index_grad_base + grad_y * grad.stride(1) + grad_x * grad.stride(2);

            sum += grad[index_grad];
        }
    }

    output[out_idx] = sum;
}

#[cube]
fn start_index<F: Float>(input_index: usize, output_size: usize, input_size: usize) -> usize {
    let numerator = F::cast_from(input_index * output_size);
    let div = (numerator / F::cast_from(input_size)).ceil();

    usize::cast_from(div)
}

#[cube]
fn end_index<F: Float>(input_index: usize, output_size: usize, input_size: usize) -> usize {
    let numerator = F::cast_from((input_index + 1) * output_size);
    let div = (numerator / F::cast_from(input_size)).ceil();
    let index = usize::cast_from(div);

    clamp_max(index, output_size)
}

pub(crate) fn interpolate_nearest_backward_launch<R: CubeRuntime>(
    out_grad: CubeTensor<R>,
    output: CubeTensor<R>,
) -> CubeTensor<R> {
    let line_size = max_line_size(&out_grad);
    let out_shape = shape_divmod(&output);
    let out_layout = linear_layout(&output, line_size);

    let working_units = output.shape.num_elements() / line_size as usize;
    let cube_dim = CubeDim::new(&out_grad.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&out_grad.client, working_units, cube_dim);

    unsafe {
        interpolate_nearest_backward_kernel::launch_unchecked(
            &out_grad.client,
            cube_count,
            cube_dim,
            address_type!(out_grad, output),
            out_grad.as_tensor_arg(line_size),
            output.as_tensor_arg(line_size),
            out_shape,
            out_layout,
            output.dtype.into(),
        )
        .expect("Kernel to never fail");
    };

    output
}
