use cubecl::std::{
    FastDivmod,
    tensor::layout::{linear::LinearLayout, *},
};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{
    CubeRuntime,
    kernel::utils::{linear_layout, shape_divmod},
    ops::max_line_size,
    tensor::CubeTensor,
};

#[cube(launch)]
fn interpolate_bicubic_kernel<F: Float>(
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

    let (rem, c) = shape_out[3].div_mod(ABSOLUTE_POS * line_size);
    let (rem, x) = shape_out[2].div_mod(rem);
    let (b, y) = shape_out[1].div_mod(rem);

    let input_height = input.shape(1) - 1;
    let output_height = clamp_min(output.shape(1) - 1, 1) as f32;
    let numerator = f32::cast_from(y * input_height);

    let frac = f32::cast_from(numerator / output_height);
    let y_in_f = frac.floor();
    let y_in = usize::cast_from(y_in_f);
    let yw = Line::empty(line_size).fill(F::cast_from(frac - y_in_f));

    let y0 = select(y_in != 0, y_in - 1, 0);
    let y1 = y_in;
    let y2 = clamp_max(y_in + 1, input_height);
    let y3 = clamp_max(y_in + 2, input_height);

    let input_width = input.shape(2) - 1;
    let output_width = clamp_min(output.shape(2) - 1, 1) as f32;
    let numerator = f32::cast_from(x * input_width);
    let frac = numerator / output_width;
    let x_in_f = frac.floor();
    let x_in = usize::cast_from(x_in_f);
    let xw = Line::empty(line_size).fill(F::cast_from(frac - x_in_f));

    let x0 = select(x_in != 0, x_in - 1, 0);
    let x1 = x_in;
    let x2 = clamp_max(x_in + 1, input_width);
    let x3 = clamp_max(x_in + 2, input_width);

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

    let inp_0 = input[index_base + y0_stride + x0_stride];
    let inp_1 = input[index_base + y0_stride + x1_stride];
    let inp_2 = input[index_base + y0_stride + x2_stride];
    let inp_3 = input[index_base + y0_stride + x3_stride];

    let coefficients0 = cubic_interp_1d::<F>(inp_0, inp_1, inp_2, inp_3, xw);

    let inp_0 = input[index_base + y1_stride + x0_stride];
    let inp_1 = input[index_base + y1_stride + x1_stride];
    let inp_2 = input[index_base + y1_stride + x2_stride];
    let inp_3 = input[index_base + y1_stride + x3_stride];

    let coefficients1 = cubic_interp_1d::<F>(inp_0, inp_1, inp_2, inp_3, xw);

    let inp_0 = input[index_base + y2_stride + x0_stride];
    let inp_1 = input[index_base + y2_stride + x1_stride];
    let inp_2 = input[index_base + y2_stride + x2_stride];
    let inp_3 = input[index_base + y2_stride + x3_stride];

    let coefficients2 = cubic_interp_1d::<F>(inp_0, inp_1, inp_2, inp_3, xw);

    let inp_0 = input[index_base + y3_stride + x0_stride];
    let inp_1 = input[index_base + y3_stride + x1_stride];
    let inp_2 = input[index_base + y3_stride + x2_stride];
    let inp_3 = input[index_base + y3_stride + x3_stride];

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
        input.as_tensor_arg(line_size),
        output.as_tensor_arg(line_size),
        out_shape,
        out_layout,
        output.dtype.into(),
    )
    .expect("Kernel to never fail");

    output
}
