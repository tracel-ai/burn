use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{FastDivmod, FastDivmodArgs},
};
use cubek::convolution::components::ConvSetupError;

use burn_backend::{
    Shape,
    ops::{DeformConvOptions, conv::calculate_conv_output_size},
};

use crate::{
    CubeRuntime,
    kernel::{
        AddOp, into_contiguous_aligned, launch_binop,
        matmul::{MatmulStrategy, matmul},
        utils::address_type,
    },
    ops::{numeric::zeros_client, reshape, swap_dims},
    tensor::CubeTensor,
};

#[derive(CubeLaunch, CubeType)]
struct DeformConv2dArgs {
    conv_stride_h: usize,
    conv_stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    padding_h: InputScalar,
    padding_w: InputScalar,
    offset_groups: usize,

    kernel_height: usize,
    kernel_width: usize,
    out_h: usize,
    out_w: usize,
}

#[cube(launch, address_type = "dynamic")]
fn deform_im2col_kernel<F: Float>(
    input: &Tensor<F>,
    offset: &Tensor<F>,
    mask: &Option<Tensor<F>>,
    columns: &mut Tensor<F>,
    pos_shape: Sequence<FastDivmod<usize>>,
    args: &DeformConv2dArgs,
    #[comptime] kernel_h_unroll: Option<usize>,
    #[comptime] kernel_w_unroll: Option<usize>,
    #[define(F)] _dtype: StorageType,
) {
    // position shape: [in_channels, batch_size, out_h, out_w]
    // columns shape: [[in_channels, kernel_h, kernel_w], [batch_size, out_h, out_w]]

    let kernel_height = kernel_h_unroll.unwrap_or(args.kernel_height);
    let unroll_h = kernel_h_unroll.is_some();
    let kernel_width = kernel_w_unroll.unwrap_or(args.kernel_width);
    let unroll_w = kernel_w_unroll.is_some();

    let out_h = args.out_h;
    let out_w = args.out_w;
    let in_channels = input.shape(1);
    let height = input.shape(2);
    let width = input.shape(3);
    let col_stride_0 = columns.stride(0);

    let (rem, out_x) = pos_shape[3].div_mod(ABSOLUTE_POS);
    let (rem, out_y) = pos_shape[2].div_mod(rem);
    let (in_channel, batch) = pos_shape[1].div_mod(rem);

    if in_channel >= in_channels {
        terminate!()
    }

    let out_k_base = in_channel * kernel_height * kernel_width;
    let out_n = batch * out_h * out_w + out_y * out_w + out_x;

    let channels_per_offset_group = in_channels / args.offset_groups;
    let group_index = in_channel / channels_per_offset_group;

    let mut col_base_idx = out_k_base * columns.stride(0) + out_n * columns.stride(1);

    let input_base_idx = batch * input.stride(0) + in_channel * input.stride(1);

    let offset_base_idx = batch * offset.stride(0)
        + group_index * kernel_height * kernel_width * 2 * offset.stride(1);

    let mask_base_idx = mask.as_ref().map(|mask| {
        batch * mask.stride(0) + group_index * kernel_height * kernel_width * mask.stride(1)
    });

    #[unroll(unroll_h)]
    for kernel_y in 0..kernel_height {
        #[unroll(unroll_w)]
        for kernel_x in 0..kernel_width {
            let mask_index = kernel_y * kernel_width + kernel_x;
            let offset_index = mask_index * 2;

            let offset_y = offset[offset_base_idx
                + offset_index * offset.stride(1)
                + out_y * offset.stride(2)
                + out_x * offset.stride(3)];
            let offset_x = offset[offset_base_idx
                + (offset_index + 1) * offset.stride(1)
                + out_y * offset.stride(2)
                + out_x * offset.stride(3)];
            let y = F::cast_from(out_y * args.conv_stride_h + kernel_y * args.dilation_h)
                - args.padding_h.get::<F>()
                + offset_y;
            let x = F::cast_from(out_x * args.conv_stride_w + kernel_x * args.dilation_w)
                - args.padding_w.get::<F>()
                + offset_x;

            let interpolated = bilinear_interpolate(input, height, width, y, x, input_base_idx);
            let value = match mask.zip::<usize>(mask_base_idx) {
                Some((mask, base_idx)) => {
                    let mask_value = mask[base_idx
                        + mask_index * mask.stride(1)
                        + out_y * mask.stride(2)
                        + out_x * mask.stride(3)];
                    mask_value * interpolated
                }
                None => interpolated,
            };

            columns[col_base_idx] = value;
            col_base_idx += col_stride_0;
        }
    }
}

#[cube]
pub(crate) fn bilinear_interpolate<F: Float>(
    input: &Tensor<F>,
    height: usize,
    width: usize,
    y: F,
    x: F,
    offset: usize,
) -> F {
    // To simplify code
    let y = f32::cast_from(y);
    let x = f32::cast_from(x);
    let stride_y = input.stride(2);
    let stride_x = input.stride(3);

    let mut result = F::new(0.0);
    if y > -1.0 && height as f32 > y && x > -1.0 && width as f32 > x {
        let y_low = y.floor();
        let x_low = x.floor();
        let y_high = (y_low + 1.) as usize;
        let x_high = (x_low + 1.) as usize;

        let zero = F::new(0.0);
        let v1: F = if y_low >= 0. && x_low >= 0. {
            input[offset + y_low as usize * stride_y + x_low as usize * stride_x]
        } else {
            zero
        };
        let v2: F = if y_low >= 0. && x_high < width {
            input[offset + y_low as usize * stride_y + x_high * stride_x]
        } else {
            zero
        };
        let v3: F = if y_high < height && x_low >= 0. {
            input[offset + y_high * stride_y + x_low as usize * stride_x]
        } else {
            zero
        };
        let v4: F = if y_high < height && x_high < width {
            input[offset + y_high * stride_y + x_high * stride_x]
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

pub(crate) fn deform_im2col<R: CubeRuntime>(
    input: CubeTensor<R>,
    offset: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    options: DeformConvOptions<2>,
    out_dims: (usize, usize),
    kernel_dims: (usize, usize),
) -> Result<CubeTensor<R>, LaunchError> {
    let client = input.client.clone();
    let device = input.device.clone();
    let dtype = input.dtype;

    let [batch_size, in_channels, _, _] = input.meta.shape().dims();
    let (out_height, out_width) = out_dims;
    let (kernel_height, kernel_width) = kernel_dims;

    let shape_out = Shape::new([
        in_channels * kernel_height * kernel_width,
        batch_size * out_height * out_width,
    ]);

    let pos_shape = [in_channels, batch_size, out_height, out_width]
        .into_iter()
        .map(|s| FastDivmodArgs::new(&client, s))
        .collect();

    let output = zeros_client(client.clone(), device.clone(), shape_out.clone(), dtype);

    let num_kernels = in_channels * batch_size * out_height * out_width;
    let cube_dim = CubeDim::new(&input.client, num_kernels);
    let cube_count = calculate_cube_count_elemwise(&input.client, num_kernels, cube_dim);

    deform_im2col_kernel::launch(
        &input.client,
        cube_count,
        cube_dim,
        address_type!(input, offset, mask, output),
        input.as_tensor_arg(1),
        offset.as_tensor_arg(1),
        mask.as_ref().map(|mask| mask.as_tensor_arg(1)).into(),
        output.as_handle_ref().as_tensor_arg(1),
        pos_shape,
        DeformConv2dArgsLaunch::new(
            ScalarArg::new(options.stride[0]),
            ScalarArg::new(options.stride[1]),
            ScalarArg::new(options.dilation[0]),
            ScalarArg::new(options.dilation[1]),
            {
                let val = options.padding[0] as f32;
                InputScalar::new(val, dtype)
            },
            {
                let val = options.padding[1] as f32;
                InputScalar::new(val, dtype)
            },
            ScalarArg::new(options.offset_groups),
            ScalarArg::new(kernel_height),
            ScalarArg::new(kernel_width),
            ScalarArg::new(out_height),
            ScalarArg::new(out_width),
        ),
        Some(kernel_height),
        Some(kernel_width),
        dtype.into(),
    );

    Ok(output)
}

pub(crate) fn deform_conv2d<R: CubeRuntime>(
    input: CubeTensor<R>,
    offset: CubeTensor<R>,
    weight: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    bias: Option<CubeTensor<R>>,
    options: DeformConvOptions<2>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let input = into_contiguous_aligned(input);
    let offset = into_contiguous_aligned(offset);
    let weight = into_contiguous_aligned(weight);
    let mask = mask.map(|it| into_contiguous_aligned(it));
    let bias = bias.map(|it| into_contiguous_aligned(it));

    let [batch_size, _, in_height, in_width] = input.meta.shape().dims();
    let [out_channels, _, kernel_h, kernel_w] = weight.meta.shape().dims();
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

    let columns = deform_im2col(input, offset, mask, options, out_dims, (kernel_h, kernel_w))?;

    let [col_size_0, col_size_1] = columns.meta.shape().dims();
    let col_size_0 = col_size_0 / groups;
    let out_c_per_group = out_channels / groups;

    let dtype = weight.dtype;
    let weight = reshape(weight, Shape::new([groups, out_c_per_group, col_size_0]));
    let columns = reshape(columns, Shape::new([groups, col_size_0, col_size_1]));
    let out = matmul(weight, columns, None, MatmulStrategy::default(), dtype)?;

    let out = reshape(out, Shape::new([out_channels, batch_size, out_h, out_w]));
    let out = swap_dims(out, 0, 1);

    if let Some(bias) = bias {
        let bias = reshape(bias, Shape::new([1, out_channels, 1, 1]));
        Ok(launch_binop::<R, AddOp>(out, bias))
    } else {
        Ok(out)
    }
}
