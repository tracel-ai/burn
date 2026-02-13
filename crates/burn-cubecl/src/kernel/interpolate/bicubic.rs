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
fn interpolate_bicubic_kernel<F: Float>(
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

    let frac = if align_corners {
        let output_height = clamp_min(output.shape(1) - 1, 1) as f32;
        (y * input_height) as f32 / output_height
    } else {
        let in_size = (input_height + 1) as f32;
        let out_size = output.shape(1) as f32;
        (y as f32 + 0.5) * (in_size / out_size) - 0.5
    };
    let y_in_f = frac.floor();
    let yw = Line::empty(line_size).fill(F::cast_from(frac - y_in_f));

    // Clamp indices in float space to handle negative coordinates from half_pixel
    let y0 = f32::clamp(y_in_f - 1.0, 0.0, input_height_f) as usize;
    let y1 = f32::clamp(y_in_f, 0.0, input_height_f) as usize;
    let y2 = f32::clamp(y_in_f + 1.0, 0.0, input_height_f) as usize;
    let y3 = f32::clamp(y_in_f + 2.0, 0.0, input_height_f) as usize;

    let input_width = input.shape(2) - 1;
    let input_width_f = input_width as f32;

    let frac = if align_corners {
        let output_width = clamp_min(output.shape(2) - 1, 1) as f32;
        (x * input_width) as f32 / output_width
    } else {
        let in_size = (input_width + 1) as f32;
        let out_size = output.shape(2) as f32;
        (x as f32 + 0.5) * (in_size / out_size) - 0.5
    };
    let x_in_f = frac.floor();
    let xw = Line::empty(line_size).fill(F::cast_from(frac - x_in_f));

    // Clamp indices in float space to handle negative coordinates from half_pixel
    let x0 = f32::clamp(x_in_f - 1.0, 0.0, input_width_f) as usize;
    let x1 = f32::clamp(x_in_f, 0.0, input_width_f) as usize;
    let x2 = f32::clamp(x_in_f + 1.0, 0.0, input_width_f) as usize;
    let x3 = f32::clamp(x_in_f + 2.0, 0.0, input_width_f) as usize;

    let index_base = b * input.stride(0) + c * input.stride(3);
    let in_stride_y = input.stride(1);
    let in_stride_x = input.stride(2);

    let y0_stride = y0 * in_stride_y;
    let y1_stride = y1 * in_stride_y;
    let y2_stride = y2 * in_stride_y;
    let y3_stride = y3 * in_stride_y;
    let x0_stride = x0 * in_stride_x;
    let x1_stride = x1 * in_stride_x;
    let x2_stride = x2 * in_stride_x;
    let x3_stride = x3 * in_stride_x;

    let inp_0 = input[(index_base + y0_stride + x0_stride) / line_size];
    let inp_1 = input[(index_base + y0_stride + x1_stride) / line_size];
    let inp_2 = input[(index_base + y0_stride + x2_stride) / line_size];
    let inp_3 = input[(index_base + y0_stride + x3_stride) / line_size];

    let coefficients0 = cubic_interp_1d::<F>(inp_0, inp_1, inp_2, inp_3, xw);

    let inp_0 = input[(index_base + y1_stride + x0_stride) / line_size];
    let inp_1 = input[(index_base + y1_stride + x1_stride) / line_size];
    let inp_2 = input[(index_base + y1_stride + x2_stride) / line_size];
    let inp_3 = input[(index_base + y1_stride + x3_stride) / line_size];

    let coefficients1 = cubic_interp_1d::<F>(inp_0, inp_1, inp_2, inp_3, xw);

    let inp_0 = input[(index_base + y2_stride + x0_stride) / line_size];
    let inp_1 = input[(index_base + y2_stride + x1_stride) / line_size];
    let inp_2 = input[(index_base + y2_stride + x2_stride) / line_size];
    let inp_3 = input[(index_base + y2_stride + x3_stride) / line_size];

    let coefficients2 = cubic_interp_1d::<F>(inp_0, inp_1, inp_2, inp_3, xw);

    let inp_0 = input[(index_base + y3_stride + x0_stride) / line_size];
    let inp_1 = input[(index_base + y3_stride + x1_stride) / line_size];
    let inp_2 = input[(index_base + y3_stride + x2_stride) / line_size];
    let inp_3 = input[(index_base + y3_stride + x3_stride) / line_size];

    let coefficients3 = cubic_interp_1d::<F>(inp_0, inp_1, inp_2, inp_3, xw);

    let val = cubic_interp_1d::<F>(
        coefficients0,
        coefficients1,
        coefficients2,
        coefficients3,
        yw,
    );

    output[out_idx] = val;
}

#[cube]
fn cubic_interp_1d<F: Float>(
    x0: Line<F>,
    x1: Line<F>,
    x2: Line<F>,
    x3: Line<F>,
    t: Line<F>,
) -> Line<F> {
    let a = lined(&x0, -0.75);

    let coeffs0 = cubic_convolution_2::<F>(t + lined(&x0, 1.0), a);
    let coeffs1 = cubic_convolution_1::<F>(t, a);
    let coeffs2 = cubic_convolution_1::<F>(lined(&x0, 1.0) - t, a);
    let coeffs3 = cubic_convolution_2::<F>(lined(&x0, 2.0) - t, a);

    x0 * coeffs0 + x1 * coeffs1 + x2 * coeffs2 + x3 * coeffs3
}

#[cube]
fn cubic_convolution_1<F: Float>(x: Line<F>, a: Line<F>) -> Line<F> {
    let conv = (a + lined(&x, 2.0)) * x;
    let tmp = a + lined(&x, 3.0);
    (conv - tmp) * x * x + lined(&x, 1.0)
}

#[cube]
fn cubic_convolution_2<F: Float>(x: Line<F>, a: Line<F>) -> Line<F> {
    let conv = a * x;
    let conv = (conv - lined(&x, 5.0) * a) * x;
    let tmp = lined(&x, 8.0) * a;
    let conv = (conv + tmp) * x;

    conv - lined(&x, 4.0) * a
}

#[cube]
fn lined<F: Float>(x: &Line<F>, #[comptime] v: f32) -> Line<F> {
    Line::empty(x.size()).fill(F::new(v))
}

pub(crate) fn interpolate_bicubic_launch<R: CubeRuntime>(
    input: CubeTensor<R>,
    output: CubeTensor<R>,
    align_corners: bool,
) -> CubeTensor<R> {
    let line_size = max_line_size(&input);
    let out_shape = shape_divmod(&output);
    let out_layout = linear_layout(&output, line_size);

    let working_units = output.shape.num_elements() / line_size as usize;
    let cube_dim = CubeDim::new(&input.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&input.client, working_units, cube_dim);

    interpolate_bicubic_kernel::launch(
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
