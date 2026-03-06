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

#[cube(launch, address_type = "dynamic")]
fn interpolate_lanczos3_kernel<F: Float>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
    shape_out: Sequence<FastDivmod<usize>>,
    out_layout: LinearLayout,
    #[comptime] align_corners: bool,
    #[define(F)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let line_size = input.line_size();
    let out_idx = out_layout.to_source_pos(ABSOLUTE_POS);

    let (rem, c) = shape_out[3].div_mod(ABSOLUTE_POS * line_size);
    let (rem, x) = shape_out[2].div_mod(rem);
    let (b, y) = shape_out[1].div_mod(rem);

    let input_height = input.shape(1) - 1;
    let input_height_f = input_height as f32;

    let y_frac = if align_corners {
        let output_height = clamp_min(output.shape(1) - 1, 1) as f32;
        (y * input_height) as f32 / output_height
    } else {
        let in_size = (input_height + 1) as f32;
        let out_size = output.shape(1) as f32;
        (y as f32 + 0.5) * (in_size / out_size) - 0.5
    };
    let y0 = f32::floor(y_frac);

    let input_width = input.shape(2) - 1;
    let input_width_f = input_width as f32;

    let x_frac = if align_corners {
        let output_width = clamp_min(output.shape(2) - 1, 1) as f32;
        (x * input_width) as f32 / output_width
    } else {
        let in_size = (input_width + 1) as f32;
        let out_size = output.shape(2) as f32;
        (x as f32 + 0.5) * (in_size / out_size) - 0.5
    };
    let x0 = f32::floor(x_frac);

    let index_base = b * input.stride(0) + c * input.stride(3);
    let in_stride_y = input.stride(1);
    let in_stride_x = input.stride(2);

    let zero = Line::empty(line_size).fill(F::new(0.0));
    let mut result = zero;

    // 6-tap separable Lanczos3 filter: ky in -2..=3, kx in -2..=3
    #[unroll]
    for ky in -2..4i32 {
        let y_pos = y0 + ky as f32;
        let y_idx = clamp(y_pos, 0.0, input_height_f) as usize;
        let wy = lanczos3_weight(y_frac - y_pos);

        #[unroll]
        for kx in -2..4i32 {
            let x_pos = x0 + kx as f32;
            let x_idx = clamp(x_pos, 0.0, input_width_f) as usize;
            let wx = lanczos3_weight(x_frac - x_pos);

            let idx = index_base + y_idx * in_stride_y + x_idx * in_stride_x;
            let pixel = input[idx / line_size];
            let w = Line::empty(line_size).fill(F::cast_from(wy * wx));
            result += pixel * w;
        }
    }

    output[out_idx] = result;
}

#[cube]
fn lanczos3_weight(x: f32) -> f32 {
    let abs_x = f32::abs(x);
    let mut result = 0.0f32;
    if abs_x < 1e-7 {
        result = 1.0;
    } else if abs_x < 3.0 {
        let pi = core::f32::consts::PI;
        let pi_x = pi * x;
        let pi_x_over_3 = pi_x / 3.0;
        result = (f32::sin(pi_x) * f32::sin(pi_x_over_3)) / (pi_x * pi_x_over_3);
    }
    result
}

pub(crate) fn interpolate_lanczos3_launch<R: CubeRuntime>(
    input: CubeTensor<R>,
    output: CubeTensor<R>,
    align_corners: bool,
) -> CubeTensor<R> {
    let line_size = max_line_size(&input);
    let out_shape = shape_divmod(&output);
    let out_layout = linear_layout(&output, line_size);

    let working_units = output.meta.num_elements() / line_size as usize;
    let cube_dim = CubeDim::new(&input.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&input.client, working_units, cube_dim);

    interpolate_lanczos3_kernel::launch(
        &input.client,
        cube_count,
        cube_dim,
        address_type!(input, output),
        input.as_tensor_arg(line_size),
        output.as_tensor_arg(line_size),
        out_shape,
        out_layout,
        align_corners,
        output.dtype.into(),
    )
    .expect("Kernel to never fail");

    output
}
