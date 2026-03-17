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
fn interpolate_lanczos3_kernel<F: Float, N: Size>(
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

    let mut result = Vector::zero();
    let mut weight_sum = 0.0f32;

    // 6-tap separable Lanczos3 filter: ky in -2..=3, kx in -2..=3
    // Skip out-of-bounds positions instead of clamping (matches TF/JAX/PIL)
    #[unroll]
    for ky in -2..4i32 {
        let y_pos = y0 + ky as f32;
        if y_pos >= 0.0 && y_pos <= input_height_f {
            let y_idx = y_pos as usize;
            let wy = lanczos3_weight(y_frac - y_pos);

            #[unroll]
            for kx in -2..4i32 {
                let x_pos = x0 + kx as f32;
                if x_pos >= 0.0 && x_pos <= input_width_f {
                    let x_idx = x_pos as usize;
                    let wx = lanczos3_weight(x_frac - x_pos);

                    let wt = wy * wx;
                    let idx = index_base + y_idx * in_stride_y + x_idx * in_stride_x;
                    let pixel = input[idx / vector_size];
                    let w = Vector::new(F::cast_from(wt));
                    result += pixel * w;
                    weight_sum += wt;
                }
            }
        }
    }

    if weight_sum != 0.0 {
        let inv_w = Vector::new(F::cast_from(1.0 / weight_sum));
        result *= inv_w;
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
    let vector_size = max_vector_size(&input);
    let out_shape = shape_divmod(&output);
    let out_layout = linear_layout(&output, vector_size);

    let working_units = output.meta.num_elements() / vector_size as usize;
    let cube_dim = CubeDim::new(&input.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&input.client, working_units, cube_dim);

    interpolate_lanczos3_kernel::launch(
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
