use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{tensor::CubeTensor, CubeRuntime, FloatElement};

#[cube(launch)]
fn interpolate_bicubic_kernel<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let batch = ABSOLUTE_POS / output.stride(0) % output.shape(0);
    let channel = ABSOLUTE_POS / output.stride(1) % output.shape(1);
    let y = ABSOLUTE_POS / output.stride(2) % output.shape(2);
    let x = ABSOLUTE_POS / output.stride(3) % output.shape(3);

    let input_height = input.shape(2) - 1;
    let output_height = F::cast_from(Max::max(output.shape(2) - 1, 1));
    let numerator = F::cast_from(y * input_height);

    let frac = numerator / output_height;
    let y_in_f = Floor::floor(frac);
    let y_in = u32::cast_from(y_in_f);
    let yw = frac - y_in_f;

    let y0 = select(y_in != 0, y_in - 1, 0);
    let y1 = y_in;
    let y2 = Min::min(y_in + 1, input_height);
    let y3 = Min::min(y_in + 2, input_height);

    let input_width = input.shape(3) - 1;
    let output_width = F::cast_from(Max::max(output.shape(3) - 1, 1));
    let numerator = F::cast_from(x * input_width);
    let frac = numerator / output_width;
    let x_in_f = Floor::floor(frac);
    let x_in = u32::cast_from(x_in_f);
    let xw = frac - x_in_f;

    let x0 = select(x_in != 0, x_in - 1, 0);
    let x1 = x_in;
    let x2 = Min::min(x_in + 1, input_width);
    let x3 = Min::min(x_in + 2, input_width);

    let index_base = batch * input.stride(0) + channel * input.stride(1);
    let in_stride_y = input.stride(2);
    let in_stride_x = input.stride(3);

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

    output[ABSOLUTE_POS] = val;
}

#[cube]
fn cubic_interp_1d<F: Float>(x0: F, x1: F, x2: F, x3: F, t: F) -> F {
    let a = F::new(-0.75);

    let coeffs0 = cubic_convolution_2::<F>(t + F::new(1.0), a);
    let coeffs1 = cubic_convolution_1::<F>(t, a);
    let coeffs2 = cubic_convolution_1::<F>(F::new(1.0) - t, a);
    let coeffs3 = cubic_convolution_2::<F>(F::new(2.0) - t, a);

    x0 * coeffs0 + x1 * coeffs1 + x2 * coeffs2 + x3 * coeffs3
}

#[cube]
fn cubic_convolution_1<F: Float>(x: F, a: F) -> F {
    let conv = (a + F::new(2.0)) * x;
    let tmp = a + F::new(3.0);
    (conv - tmp) * x * x + F::new(1.0)
}

#[cube]
fn cubic_convolution_2<F: Float>(x: F, a: F) -> F {
    let conv = a * x;
    let conv = (conv - F::new(5.0) * a) * x;
    let tmp = F::new(8.0) * a;
    let conv = (conv + tmp) * x;

    conv - F::new(4.0) * a
}

pub(crate) fn interpolate_bicubic_launch<R: CubeRuntime, E: FloatElement>(
    input: CubeTensor<R>,
    output: CubeTensor<R>,
) -> CubeTensor<R> {
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(output.shape.num_elements(), cube_dim);

    interpolate_bicubic_kernel::launch::<E, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg::<E>(1),
        output.as_tensor_arg::<E>(1),
    );

    output
}
