use cubecl::{calculate_cube_count_elemwise, prelude::*};

use burn_tensor::{
    ops::{conv::calculate_conv_output_size, DeformConvOptions},
    Shape,
};

use crate::{
    kernel::{
        into_contiguous, launch_binop,
        matmul::{matmul, MatmulStrategy},
        AddOp,
    },
    ops::{
        numeric::{ones_device, zeros_device},
        reshape, swap_dims,
    },
    tensor::CubeTensor,
    CubeRuntime, FloatElement,
};

use super::ConvLaunchError;

#[derive(CubeLaunch)]
struct DeformConv2dArgs<F: Float> {
    conv_stride_h: u32,
    conv_stride_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    padding_h: F,
    padding_w: F,
    offset_groups: u32,

    kernel_height: u32,
    kernel_width: u32,
    out_h: u32,
    out_w: u32,

    col_stride_0: u32,
}

#[cube(launch)]
fn deform_im2col_kernel<F: Float>(
    input: &Tensor<F>,
    offset: &Tensor<F>,
    mask: &Tensor<F>,
    columns: &mut Tensor<F>,
    args: &DeformConv2dArgs<F>,
    #[comptime] kernel_h_unroll: Option<u32>,
    #[comptime] kernel_w_unroll: Option<u32>,
    #[comptime] use_mask: bool,
) {
    // position shape: [in_channels, batch_size, out_h, out_w]
    // columns shape: [[in_channels, kernel_h, kernel_w], [batch_size, out_h, out_w]]

    let kernel_height = kernel_h_unroll.unwrap_or(args.kernel_height);
    let unroll_h = kernel_h_unroll.is_some();
    let kernel_width = kernel_w_unroll.unwrap_or(args.kernel_width);
    let unroll_w = kernel_w_unroll.is_some();

    // Keep mask in bind group
    let default_mask_value = mask[0];

    let out_h = args.out_h;
    let out_w = args.out_w;
    let batch_size = input.shape(0);
    let in_channels = input.shape(1);
    let height = input.shape(2);
    let width = input.shape(3);
    let col_stride_0 = args.col_stride_0;

    let out_x = ABSOLUTE_POS % out_w;
    let out_y = (ABSOLUTE_POS / out_w) % out_h;
    let out_batch = (ABSOLUTE_POS / (out_w * out_h)) % batch_size;
    let in_channel = ABSOLUTE_POS / (out_w * out_h * batch_size);
    let out_channel = in_channel * kernel_height * kernel_width;

    let channels_per_offset_group = in_channels / args.offset_groups;
    let group_index = in_channel / channels_per_offset_group;

    let mut col_base_idx =
        out_channel * col_stride_0 + out_batch * (out_h * out_w) + out_y * out_w + out_x;

    let input_base_idx = out_batch * input.stride(0) + in_channel * input.stride(1);

    let offset_base_idx = out_batch * offset.stride(0)
        + group_index * kernel_height * kernel_width * 2 * out_h * out_w;
    let mut mask_base_idx = 0;
    if use_mask {
        mask_base_idx =
            out_batch * mask.stride(0) + group_index * kernel_height * kernel_width * out_h * out_w;
    }

    #[unroll(unroll_h)]
    for kernel_y in 0..kernel_height {
        #[unroll(unroll_w)]
        for kernel_x in 0..kernel_width {
            let mask_index = kernel_y * kernel_width + kernel_x;
            let offset_index = mask_index * 2;

            let mut mask_value = default_mask_value;
            if use_mask {
                mask_value = mask[mask_base_idx
                    + mask_index * mask.stride(1)
                    + out_y * mask.stride(2)
                    + out_x * mask.stride(3)];
            }

            let offset_y = offset[offset_base_idx
                + offset_index * offset.stride(1)
                + out_y * offset.stride(2)
                + out_x * offset.stride(3)];
            let offset_x = offset[offset_base_idx
                + (offset_index + 1) * offset.stride(1)
                + out_y * offset.stride(2)
                + out_x * offset.stride(3)];
            let y = F::cast_from(out_y * args.conv_stride_h + kernel_y * args.dilation_h)
                - args.padding_h
                + offset_y;
            let x = F::cast_from(out_x * args.conv_stride_w + kernel_x * args.dilation_w)
                - args.padding_w
                + offset_x;

            let interpolated = bilinear_interpolate(input, height, width, y, x, input_base_idx);

            columns[col_base_idx] = mask_value * interpolated;
            col_base_idx += col_stride_0;
        }
    }
}

#[cube]
pub(crate) fn bilinear_interpolate<F: Float>(
    input: &Tensor<F>,
    height: u32,
    width: u32,
    y: F,
    x: F,
    offset: u32,
) -> F {
    // To simplify code
    let y = f32::cast_from(y);
    let x = f32::cast_from(x);

    let mut result = F::new(0.0);
    if y > -1.0 && height as f32 > y && x > -1.0 && width as f32 > x {
        let in_w = u32::cast_from(width);

        let y_low = f32::floor(y);
        let x_low = f32::floor(x);
        let y_high = (y_low + 1.) as u32;
        let x_high = (x_low + 1.) as u32;

        let zero = F::new(0.0);
        let v1: F = if y_low >= 0. && x_low >= 0. {
            input[offset + y_low as u32 * in_w + x_low as u32]
        } else {
            zero
        };
        let v2: F = if y_low >= 0. && x_high < width {
            input[offset + y_low as u32 * in_w + x_high]
        } else {
            zero
        };
        let v3: F = if y_high < height && x_low >= 0. {
            input[offset + y_high * in_w + x_low as u32]
        } else {
            zero
        };
        let v4: F = if y_high < height && x_high < width {
            input[offset + y_high * in_w + x_high]
        } else {
            zero
        };

        let l_y = y - y_low;
        let l_x = x - x_low;
        let h_y = 1.0 - l_y;
        let h_x = 1.0 - l_x;

        let w1 = F::cast_from(h_y * h_x);
        let w2 = F::cast_from(h_y * l_x);
        let w3 = F::cast_from(l_y * h_x);
        let w4 = F::cast_from(l_y * l_x);

        result = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
    }
    result
}

pub(crate) fn deform_im2col<R: CubeRuntime, E: FloatElement>(
    input: CubeTensor<R>,
    offset: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    options: DeformConvOptions<2>,
    out_dims: (usize, usize),
    kernel_dims: (usize, usize),
) -> CubeTensor<R> {
    let client = input.client.clone();
    let device = input.device.clone();

    let [batch_size, in_channels, _, _] = input.shape.dims();
    let (out_height, out_width) = out_dims;
    let (kernel_height, kernel_width) = kernel_dims;

    let shape_out = Shape::new([
        in_channels * kernel_height * kernel_width,
        batch_size * out_height * out_width,
    ]);

    let output = zeros_device::<R, E>(client.clone(), device.clone(), shape_out.clone());
    let use_mask = mask.is_some();
    let mask = mask.unwrap_or_else(|| {
        ones_device::<R, E>(
            client.clone(),
            device.clone(),
            Shape::new([
                offset.shape.dims[0],
                offset.shape.dims[1] / 2,
                offset.shape.dims[2],
                offset.shape.dims[3],
            ]),
        )
    });

    let num_kernels = in_channels * batch_size * out_height * out_width;
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_kernels, cube_dim);

    deform_im2col_kernel::launch::<E, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_handle_ref().as_tensor_arg(1),
        offset.as_handle_ref().as_tensor_arg(1),
        mask.as_handle_ref().as_tensor_arg(1),
        output.as_handle_ref().as_tensor_arg(1),
        DeformConv2dArgsLaunch::new(
            ScalarArg::new(options.stride[0] as u32),
            ScalarArg::new(options.stride[1] as u32),
            ScalarArg::new(options.dilation[0] as u32),
            ScalarArg::new(options.dilation[1] as u32),
            ScalarArg::new(E::from_elem(options.padding[0] as f32)),
            ScalarArg::new(E::from_elem(options.padding[1] as f32)),
            ScalarArg::new(options.offset_groups as u32),
            ScalarArg::new(kernel_height as u32),
            ScalarArg::new(kernel_width as u32),
            ScalarArg::new(out_height as u32),
            ScalarArg::new(out_width as u32),
            ScalarArg::new(output.strides[0] as u32),
        ),
        Some(kernel_height as u32),
        Some(kernel_width as u32),
        use_mask,
    );

    output
}

pub(crate) fn deform_conv2d<R: CubeRuntime, E: FloatElement>(
    input: CubeTensor<R>,
    offset: CubeTensor<R>,
    weight: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    bias: Option<CubeTensor<R>>,
    options: DeformConvOptions<2>,
) -> Result<CubeTensor<R>, ConvLaunchError> {
    let input = into_contiguous(input);
    let offset = into_contiguous(offset);
    let weight = into_contiguous(weight);
    let mask = mask.map(|it| into_contiguous(it));
    let bias = bias.map(|it| into_contiguous(it));

    let [batch_size, _, in_height, in_width] = input.shape.dims();
    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims();
    let groups = options.weight_groups;

    let out_h = calculate_conv_output_size(
        kernel_h,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        in_height,
    );
    let out_w = calculate_conv_output_size(
        kernel_w,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
        in_width,
    );
    let out_dims = (out_h, out_w);

    let columns =
        deform_im2col::<R, E>(input, offset, mask, options, out_dims, (kernel_h, kernel_w));

    let [col_size_0, col_size_1] = columns.shape.dims();
    let col_size_0 = col_size_0 / groups;
    let out_c_per_group = out_channels / groups;

    let weight = reshape(weight, Shape::new([groups, out_c_per_group, col_size_0]));
    let columns = reshape(columns, Shape::new([groups, col_size_0, col_size_1]));
    let out = matmul::<R, E>(weight, columns, None, MatmulStrategy::default())?;

    let out = reshape(out, Shape::new([out_channels, batch_size, out_h, out_w]));
    let out = swap_dims(out, 0, 1);

    if let Some(bias) = bias {
        let bias = reshape(bias, Shape::new([1, out_channels, 1, 1]));
        Ok(launch_binop::<R, E, AddOp>(out, bias))
    } else {
        Ok(out)
    }
}
