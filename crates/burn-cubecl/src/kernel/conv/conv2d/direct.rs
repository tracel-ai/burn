use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    Shape,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{
    kernel::{conv::ConvLaunchError, into_contiguous},
    ops::{
        numeric::{empty_device, zeros_device},
        reshape,
    },
    tensor::JitTensor,
    FloatElement, JitRuntime,
};

#[derive(CubeLaunch)]
struct Conv2dArgs {
    conv_stride_0: u32,
    conv_stride_1: u32,
    dilation_0: u32,
    dilation_1: u32,
    padding_0: u32,
    padding_1: u32,
    channels_per_group: u32,
}

#[cube(launch)]
fn direct_conv2d_kernel<F: Float>(
    input: &Tensor<F>,
    weight: &Tensor<F>,
    bias: &Tensor<F>,
    output: &mut Tensor<F>,
    args: &Conv2dArgs,
    #[comptime] kernel_size_1_unroll: Option<u32>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let in_channels = weight.shape(1);

    let kernel_size_0 = weight.shape(2);
    let kernel_size_1 = kernel_size_1_unroll.unwrap_or_else(|| weight.shape(3));
    let unroll_1 = kernel_size_1_unroll.is_some();

    let b = ABSOLUTE_POS / output.stride(0) % output.shape(0);
    let oc = ABSOLUTE_POS / output.stride(1) % output.shape(1);
    let oh = ABSOLUTE_POS / output.stride(2) % output.shape(2);
    let ow = ABSOLUTE_POS / output.stride(3) % output.shape(3);

    let g = oc / args.channels_per_group;
    let ic_start = in_channels * g;
    let ic_end = ic_start + in_channels;
    let mut sum = bias[oc];

    let ih_base = oh * args.conv_stride_0;
    let iw_base = ow * args.conv_stride_1;

    let weight_stride_1 = weight.stride(1);
    let weight_stride_2 = weight.stride(2);
    let weight_stride_3 = weight.stride(3);

    let input_stride_1 = input.stride(1);
    let input_stride_2 = input.stride(2);
    let input_stride_3 = input.stride(3);
    let input_shape_2 = input.shape(2);
    let input_shape_3 = input.shape(3);

    let border_top = args.padding_0;
    let border_left = args.padding_1;
    let border_bottom = input_shape_2 + args.padding_0;
    let border_right = input_shape_3 + args.padding_1;

    let index_input_0 = b * input.stride(0);
    let index_weight_0 = oc * weight.stride(0);

    for ic in ic_start..ic_end {
        let index_input_1 = ic * input_stride_1;
        let index_weight_1 = (ic - ic_start) * weight_stride_1;

        for kh in 0..kernel_size_0 {
            #[unroll(unroll_1)]
            for kw in 0..kernel_size_1 {
                let ih = kh * args.dilation_0 + ih_base;
                let iw = kw * args.dilation_1 + iw_base;

                let within_padding = ih >= border_top
                    && ih < border_bottom
                    && iw >= border_left
                    && iw < border_right;

                if within_padding {
                    let ih_pad = ih - args.padding_0;
                    let iw_pad = iw - args.padding_1;

                    let index_input = index_input_0
                        + index_input_1
                        + ih_pad * input_stride_2
                        + iw_pad * input_stride_3;

                    let index_weight = index_weight_0
                        + index_weight_1
                        + kh * weight_stride_2
                        + kw * weight_stride_3;

                    sum += input[index_input] * weight[index_weight];
                }
            }
        }
    }

    output[ABSOLUTE_POS] = sum;
}

/// Perform a 2D convolution using the direct convolution algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
pub fn conv2d_direct<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    options: ConvOptions<2>,
) -> Result<JitTensor<R>, ConvLaunchError> {
    let [batch_size, _, in_height, in_width] = input.shape.dims();
    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims();
    let channels_per_group = out_channels / options.groups;

    // Limit loop unrolling factor to 8 or smaller
    let kernel_w_unroll = (kernel_w <= 8).then_some(kernel_w as u32);

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

    let input = into_contiguous(input);
    let weight = into_contiguous(weight);

    let shape_out = Shape::new([batch_size, out_channels, out_h, out_w]);
    let output = empty_device::<R, E>(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
    );

    let bias = match bias {
        Some(bias) => {
            let shape = Shape::from([bias.shape.dims[0], 1, 1, 1]);
            reshape(bias, shape)
        }
        None => {
            let shape = Shape::from([output.shape.dims[0], 1, 1, 1]);
            zeros_device::<R, E>(input.client.clone(), input.device.clone(), shape)
        }
    };

    let num_elems_output = output.shape.num_elements();
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems_output, cube_dim);

    direct_conv2d_kernel::launch::<E, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg::<E>(1),
        weight.as_tensor_arg::<E>(1),
        bias.as_tensor_arg::<E>(1),
        output.as_tensor_arg::<E>(1),
        Conv2dArgsLaunch::new(
            ScalarArg::new(options.stride[0] as u32),
            ScalarArg::new(options.stride[1] as u32),
            ScalarArg::new(options.dilation[0] as u32),
            ScalarArg::new(options.dilation[1] as u32),
            ScalarArg::new(options.padding[0] as u32),
            ScalarArg::new(options.padding[1] as u32),
            ScalarArg::new(channels_per_group as u32),
        ),
        kernel_w_unroll,
    );

    Ok(output)
}
