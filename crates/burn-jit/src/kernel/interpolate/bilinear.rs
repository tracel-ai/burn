use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{tensor::JitTensor, FloatElement, JitRuntime};

#[cube(launch)]
fn interpolate_bilinear_kernel<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }

    let batch = ABSOLUTE_POS / output.stride(0) % output.shape(0);
    let channel = ABSOLUTE_POS / output.stride(1) % output.shape(1);
    let y = ABSOLUTE_POS / output.stride(2) % output.shape(2);
    let x = ABSOLUTE_POS / output.stride(3) % output.shape(3);

    let numerator = F::cast_from(input.shape(2) - 1);
    let denominator = F::cast_from(Max::max(output.shape(2) - 1, 1));
    let factor = F::cast_from(y);

    let frac = factor * numerator / denominator;

    let v0 = Floor::floor(frac);
    let v1: F = Ceil::ceil(frac);
    let yw = frac - v0;
    let yw_ = F::new(1.0) - yw;
    let y0 = u32::cast_from(v0);
    let y1 = u32::cast_from(v1);

    let numerator = F::cast_from(input.shape(3) - 1);
    let denominator = F::cast_from(Max::max(output.shape(3) - 1, 1));
    let factor = F::cast_from(x);
    let frac = factor * numerator / denominator;
    let v0 = Floor::floor(frac);
    let v1: F = Ceil::ceil(frac);
    let xw = frac - v0;
    let xw_ = F::new(1.0) - xw;
    let x0 = u32::cast_from(v0);
    let x1 = u32::cast_from(v1);

    let index_base = batch * input.stride(0) + channel * input.stride(1);

    let in_stride_y = input.stride(2);
    let in_stride_x = input.stride(3);

    let y0_stride = y0 * in_stride_y;
    let y1_stride = y1 * in_stride_y;
    let x0_stride = x0 * in_stride_x;
    let x1_stride = x1 * in_stride_x;

    let p_a = input[index_base + y0_stride + x0_stride] * xw_ * yw_;
    let p_b = input[index_base + y0_stride + x1_stride] * xw * yw_;
    let p_c = input[index_base + y1_stride + x0_stride] * xw_ * yw;
    let p_d = input[index_base + y1_stride + x1_stride] * xw * yw;

    output[ABSOLUTE_POS] = p_a + p_b + p_c + p_d;
}

pub(crate) fn interpolate_bilinear_launch<R: JitRuntime, F: FloatElement>(
    input: JitTensor<R, F>,
    output: JitTensor<R, F>,
) -> JitTensor<R, F> {
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(output.shape.num_elements(), cube_dim);

    interpolate_bilinear_kernel::launch::<F, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(1),
        output.as_tensor_arg(1),
    );

    output
}
