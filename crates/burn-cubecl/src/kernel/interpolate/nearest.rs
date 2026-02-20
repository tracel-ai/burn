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
fn interpolate_nearest_kernel<F: Float>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    shape_out: Sequence<FastDivmod<usize>>,
    out_layout: LinearLayout,
    #[define(F)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let line_size = input.line_size();
    let out_idx = out_layout.to_source_pos(ABSOLUTE_POS);

    let out_pos = ABSOLUTE_POS * line_size;

    let (h_in, w_in) = (input.shape(1) as f32, input.shape(2) as f32);
    let (h_out, w_out) = (output.shape(1) as f32, output.shape(2) as f32);

    let (rem, c) = shape_out[3].div_mod(out_pos);
    let (rem, x) = shape_out[2].div_mod(rem);
    let (b, y) = shape_out[1].div_mod(rem);

    let y = y as f32 * (h_in / h_out);
    let x = x as f32 * (w_in / w_out);

    let in_idx = b * input.stride(0)
        + y as usize * input.stride(1)
        + x as usize * input.stride(2)
        + c * input.stride(3);

    output[out_idx] = input[in_idx / line_size];
}

pub(crate) fn interpolate_nearest_launch<R: CubeRuntime>(
    input: CubeTensor<R>,
    output: CubeTensor<R>,
) -> CubeTensor<R> {
    let client = input.client.clone();

    let line_size = max_line_size(&input);

    let working_units = output.meta.num_elements() / line_size as usize;
    let cube_dim = CubeDim::new(&input.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&input.client, working_units, cube_dim);

    let shape_out = shape_divmod(&output);
    let out_layout = linear_layout(&output, line_size);

    unsafe {
        interpolate_nearest_kernel::launch_unchecked(
            &client,
            cube_count,
            cube_dim,
            address_type!(input, output),
            input.as_tensor_arg(line_size),
            output.as_tensor_arg(line_size),
            shape_out,
            out_layout,
            output.dtype.into(),
        )
        .expect("Kernel to never fail");
    };

    output
}
