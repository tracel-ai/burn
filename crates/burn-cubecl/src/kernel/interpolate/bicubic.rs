use cubecl::std::{
    FastDivmod,
    tensor::layout::{linear::LinearLayout, *},
};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{
    CubeRuntime,
    kernel::utils::{address_type, linear_layout, shape_divmod},
    ops::max_vector_size,
    tensor::CubeTensor,
};

#[cube(launch, address_type = "dynamic")]
fn interpolate_bicubic_kernel<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    shape_out: Sequence<FastDivmod<usize>>,
    out_layout: LinearLayout,
    #[comptime] align_corners: bool,
    #[define(F)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let vector_size = input.vector_size();
    let out_idx = out_layout.to_source_pos(ABSOLUTE_POS);

    let (rem, c) = shape_out[3].div_mod(ABSOLUTE_POS * vector_size);
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
    let yw = Vector::new(F::cast_from(frac - y_in_f));

    // Clamp indices in float space to handle negative coordinates from half_pixel
    let y0 = clamp(y_in_f - 1.0, 0.0, input_height_f) as usize;
    let y1 = clamp(y_in_f, 0.0, input_height_f) as usize;
    let y2 = clamp(y_in_f + 1.0, 0.0, input_height_f) as usize;
    let y3 = clamp(y_in_f + 2.0, 0.0, input_height_f) as usize;

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
    let xw = Vector::new(F::cast_from(frac - x_in_f));

    // Clamp indices in float space to handle negative coordinates from half_pixel
    let x0 = clamp(x_in_f - 1.0, 0.0, input_width_f) as usize;
    let x1 = clamp(x_in_f, 0.0, input_width_f) as usize;
    let x2 = clamp(x_in_f + 1.0, 0.0, input_width_f) as usize;
    let x3 = clamp(x_in_f + 2.0, 0.0, input_width_f) as usize;

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

    let inp_0 = input[(index_base + y0_stride + x0_stride) / vector_size];
    let inp_1 = input[(index_base + y0_stride + x1_stride) / vector_size];
    let inp_2 = input[(index_base + y0_stride + x2_stride) / vector_size];
    let inp_3 = input[(index_base + y0_stride + x3_stride) / vector_size];

    let coefficients0 = cubic_interp_1d(inp_0, inp_1, inp_2, inp_3, xw);

    let inp_0 = input[(index_base + y1_stride + x0_stride) / vector_size];
    let inp_1 = input[(index_base + y1_stride + x1_stride) / vector_size];
    let inp_2 = input[(index_base + y1_stride + x2_stride) / vector_size];
    let inp_3 = input[(index_base + y1_stride + x3_stride) / vector_size];

    let coefficients1 = cubic_interp_1d(inp_0, inp_1, inp_2, inp_3, xw);

    let inp_0 = input[(index_base + y2_stride + x0_stride) / vector_size];
    let inp_1 = input[(index_base + y2_stride + x1_stride) / vector_size];
    let inp_2 = input[(index_base + y2_stride + x2_stride) / vector_size];
    let inp_3 = input[(index_base + y2_stride + x3_stride) / vector_size];

    let coefficients2 = cubic_interp_1d(inp_0, inp_1, inp_2, inp_3, xw);

    let inp_0 = input[(index_base + y3_stride + x0_stride) / vector_size];
    let inp_1 = input[(index_base + y3_stride + x1_stride) / vector_size];
    let inp_2 = input[(index_base + y3_stride + x2_stride) / vector_size];
    let inp_3 = input[(index_base + y3_stride + x3_stride) / vector_size];

    let coefficients3 = cubic_interp_1d(inp_0, inp_1, inp_2, inp_3, xw);

    let val = cubic_interp_1d(
        coefficients0,
        coefficients1,
        coefficients2,
        coefficients3,
        yw,
    );

    output[out_idx] = val;
}

#[cube]
fn cubic_interp_1d<F: Float, N: Size>(
    x0: Vector<F, N>,
    x1: Vector<F, N>,
    x2: Vector<F, N>,
    x3: Vector<F, N>,
    t: Vector<F, N>,
) -> Vector<F, N> {
    let a = float(-0.75);

    let coeffs0 = cubic_convolution_2(t + float(1.0), a);
    let coeffs1 = cubic_convolution_1(t, a);
    let coeffs2 = cubic_convolution_1(float(1.0) - t, a);
    let coeffs3 = cubic_convolution_2(float(2.0) - t, a);

    x0 * coeffs0 + x1 * coeffs1 + x2 * coeffs2 + x3 * coeffs3
}

#[cube]
fn cubic_convolution_1<F: Float, N: Size>(x: Vector<F, N>, a: Vector<F, N>) -> Vector<F, N> {
    let conv = (a + float(2.0)) * x;
    let tmp = a + float(3.0);
    (conv - tmp) * x * x + float(1.0)
}

#[cube]
fn cubic_convolution_2<F: Float, N: Size>(x: Vector<F, N>, a: Vector<F, N>) -> Vector<F, N> {
    let conv = a * x;
    let conv = (conv - float(5.0) * a) * x;
    let tmp = float(8.0) * a;
    let conv = (conv + tmp) * x;

    conv - float(4.0) * a
}

#[cube]
fn float<F: Float, N: Size>(#[comptime] v: f32) -> Vector<F, N> {
    Vector::new(F::new(v))
}

pub(crate) fn interpolate_bicubic_launch<R: CubeRuntime>(
    input: CubeTensor<R>,
    output: CubeTensor<R>,
    align_corners: bool,
) -> CubeTensor<R> {
    let vector_size = max_vector_size(&input);
    let out_shape = shape_divmod(&output);
    let out_layout = linear_layout(&output, vector_size);

    let working_units = output.meta.num_elements() / vector_size as usize;
    let cube_dim = CubeDim::new(&input.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&input.client, working_units, cube_dim);

    interpolate_bicubic_kernel::launch(
        &output.client,
        cube_count,
        cube_dim,
        address_type!(input, output),
        vector_size,
        input.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        out_shape,
        out_layout,
        align_corners,
        output.dtype.into(),
    );

    output
}
