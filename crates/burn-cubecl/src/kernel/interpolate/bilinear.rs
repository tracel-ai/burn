use cubecl::{calculate_cube_count_elemwise, prelude::*};
use cubecl::{
    num_traits::Zero,
    std::{
        FastDivmod,
        tensor::layout::{linear::LinearLayout, *},
    },
};

use crate::{
    CubeRuntime,
    kernel::utils::{address_type, linear_layout, shape_divmod},
    ops::max_vector_size,
    tensor::CubeTensor,
};

#[cube(launch, address_type = "dynamic")]
fn interpolate_bilinear_kernel<F: Float, N: Size>(
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

    let frac = if align_corners {
        let numerator = (input.shape(1) - 1) as f32;
        let denominator = clamp_min(output.shape(1) - 1, 1) as f32;
        y as f32 * (numerator / denominator)
    } else {
        let in_size = input.shape(1) as f32;
        let out_size = output.shape(1) as f32;
        clamp(
            (y as f32 + 0.5) * (in_size / out_size) - 0.5,
            0.0,
            in_size - 1.0,
        )
    };

    let v0 = frac.floor();
    let v1 = frac.ceil();
    let yw = F::cast_from(frac - v0);
    let yw_ = Vector::new(F::one() - yw);
    let yw = Vector::new(yw);
    let y0_ok = v0 >= 0.0;
    let y0 = v0 as usize;
    let y1 = v1 as usize;

    let frac = if align_corners {
        let numerator = (input.shape(2) - 1) as f32;
        let denominator = clamp_min(output.shape(2) - 1, 1) as f32;
        x as f32 * (numerator / denominator)
    } else {
        let in_size = input.shape(2) as f32;
        let out_size = output.shape(2) as f32;
        clamp(
            (x as f32 + 0.5) * (in_size / out_size) - 0.5,
            0.0,
            in_size - 1.0,
        )
    };
    let v0 = frac.floor();
    let v1 = frac.ceil();
    let xw = F::cast_from(frac - v0);
    let xw_ = Vector::new(F::one() - xw);
    let xw = Vector::new(xw);
    let x0_ok = v0 >= 0.0;
    let x0 = v0 as usize;
    let x1 = v1 as usize;

    let index_base = b * input.stride(0) + c * input.stride(3);

    let in_stride_y = input.stride(1);
    let in_stride_x = input.stride(2);

    let y0_stride = y0 * in_stride_y;
    let y1_stride = y1 * in_stride_y;
    let x0_stride = x0 * in_stride_x;
    let x1_stride = x1 * in_stride_x;

    let height = input.shape(1);
    let width = input.shape(2);

    let y1_ok = y1 < height;
    let x1_ok = x1 < width;

    let zero = Vector::zero();

    let p_a = select(
        x0_ok && y0_ok,
        input[(index_base + y0_stride + x0_stride) / vector_size] * xw_ * yw_,
        zero,
    );
    let p_b = select(
        x1_ok && y0_ok,
        input[(index_base + y0_stride + x1_stride) / vector_size] * xw * yw_,
        zero,
    );
    let p_c = select(
        x0_ok && y1_ok,
        input[(index_base + y1_stride + x0_stride) / vector_size] * xw_ * yw,
        zero,
    );
    let p_d = select(
        x1_ok && y1_ok,
        input[(index_base + y1_stride + x1_stride) / vector_size] * xw * yw,
        zero,
    );

    output[out_idx] = p_a + p_b + p_c + p_d;
}

pub(crate) fn interpolate_bilinear_launch<R: CubeRuntime>(
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

    interpolate_bilinear_kernel::launch(
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
