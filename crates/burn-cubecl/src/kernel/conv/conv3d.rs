use cubecl::{calculate_cube_count_elemwise, prelude::*};

use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    Shape,
};

use crate::{
    kernel::into_contiguous,
    ops::{
        numeric::{empty_device, zeros_device},
        reshape,
    },
    tensor::CubeTensor,
    CubeRuntime, FloatElement,
};

#[derive(CubeLaunch)]
struct Conv3dArgs {
    conv_stride_0: u32,
    conv_stride_1: u32,
    conv_stride_2: u32,
    dilation_0: u32,
    dilation_1: u32,
    dilation_2: u32,
    padding_0: u32,
    padding_1: u32,
    padding_2: u32,
    groups: u32,
}

#[cube(launch)]
fn conv3d_kernel<F: Float>(
    input: Tensor<F>,
    weight: Tensor<F>,
    bias: Tensor<F>,
    mut output: Tensor<F>,
    args: Conv3dArgs,
    #[comptime] kernel_size_0_unroll: Option<u32>,
    #[comptime] kernel_size_1_unroll: Option<u32>,
    #[comptime] kernel_size_2_unroll: Option<u32>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let in_channels = weight.shape(1);

    let kernel_size_0 = kernel_size_0_unroll.unwrap_or_else(|| weight.shape(2));
    let unroll_0 = kernel_size_0_unroll.is_some();
    let kernel_size_1 = kernel_size_1_unroll.unwrap_or_else(|| weight.shape(3));
    let unroll_1 = kernel_size_1_unroll.is_some();
    let kernel_size_2 = kernel_size_2_unroll.unwrap_or_else(|| weight.shape(4));
    let unroll_2 = kernel_size_2_unroll.is_some();

    let b = ABSOLUTE_POS / output.stride(0) % output.shape(0);
    let oc = ABSOLUTE_POS / output.stride(1) % output.shape(1);
    let od = ABSOLUTE_POS / output.stride(2) % output.shape(2);
    let oh = ABSOLUTE_POS / output.stride(3) % output.shape(3);
    let ow = ABSOLUTE_POS / output.stride(4) % output.shape(4);

    let g = (weight.shape(0) + oc) % args.groups;
    let ic_start = in_channels * g;
    let ic_end = ic_start + in_channels;
    let mut sum = bias[oc];

    let id_base = od * args.conv_stride_0;
    let ih_base = oh * args.conv_stride_1;
    let iw_base = ow * args.conv_stride_2;

    let weight_stride_1 = weight.stride(1);
    let weight_stride_2 = weight.stride(2);
    let weight_stride_3 = weight.stride(3);
    let weight_stride_4 = weight.stride(4);

    let input_stride_1 = input.stride(1);
    let input_stride_2 = input.stride(2);
    let input_stride_3 = input.stride(3);
    let input_stride_4 = input.stride(4);
    let input_shape_2 = input.shape(2);
    let input_shape_3 = input.shape(3);
    let input_shape_4 = input.shape(4);

    let border_front = args.padding_0;
    let border_top = args.padding_1;
    let border_left = args.padding_2;
    let border_back = input_shape_2 + args.padding_0;
    let border_bottom = input_shape_3 + args.padding_1;
    let border_right = input_shape_4 + args.padding_2;

    let index_input_0 = b * input.stride(0);
    let index_weight_0 = oc * weight.stride(0);

    for ic in ic_start..ic_end {
        let index_input_1 = ic * input_stride_1;
        let index_weight_1 = (ic - ic_start) * weight_stride_1;

        #[unroll(unroll_0)]
        for kd in 0..kernel_size_0 {
            #[unroll(unroll_1)]
            for kh in 0..kernel_size_1 {
                #[unroll(unroll_2)]
                for kw in 0..kernel_size_2 {
                    let id = kd * args.dilation_0 + id_base;
                    let ih = kh * args.dilation_1 + ih_base;
                    let iw = kw * args.dilation_2 + iw_base;

                    let within_padding = id >= border_front
                        && id < border_back
                        && ih >= border_top
                        && ih < border_bottom
                        && iw >= border_left
                        && iw < border_right;

                    if within_padding {
                        let id_pad = id - args.padding_0;
                        let ih_pad = ih - args.padding_1;
                        let iw_pad = iw - args.padding_2;

                        let index_input = index_input_0
                            + index_input_1
                            + id_pad * input_stride_2
                            + ih_pad * input_stride_3
                            + iw_pad * input_stride_4;

                        let index_weight = index_weight_0
                            + index_weight_1
                            + kd * weight_stride_2
                            + kh * weight_stride_3
                            + kw * weight_stride_4;

                        sum += input[index_input] * weight[index_weight];
                    }
                }
            }
        }
    }

    output[ABSOLUTE_POS] = sum;
}

pub(crate) fn conv3d<R: CubeRuntime, E: FloatElement>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<3>,
) -> CubeTensor<R> {
    let input = into_contiguous(input);
    let weight = into_contiguous(weight);
    let [batch_size, _, in_depth, in_height, in_width] = input.shape.dims();
    let [out_channels, _, kernel_0, kernel_1, kernel_2] = weight.shape.dims();

    let out_0 = calculate_conv_output_size(
        kernel_0,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        in_depth,
    );
    let out_1 = calculate_conv_output_size(
        kernel_1,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
        in_height,
    );
    let out_2 = calculate_conv_output_size(
        kernel_2,
        options.stride[2],
        options.padding[2],
        options.dilation[2],
        in_width,
    );

    let shape_out = Shape::new([batch_size, out_channels, out_0, out_1, out_2]);

    let output = empty_device::<R, E>(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
    );

    let bias = match bias {
        Some(bias) => {
            let shape = Shape::from([bias.shape.dims[0], 1, 1, 1, 1]);
            reshape(bias, shape)
        }
        None => {
            let shape = Shape::from([output.shape.dims[0], 1, 1, 1, 1]);
            zeros_device::<R, E>(input.client.clone(), input.device.clone(), shape)
        }
    };

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(output.shape.num_elements(), cube_dim);

    conv3d_kernel::launch::<E, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg::<E>(1),
        weight.as_tensor_arg::<E>(1),
        bias.as_tensor_arg::<E>(1),
        output.as_tensor_arg::<E>(1),
        Conv3dArgsLaunch::new(
            ScalarArg::new(options.stride[0] as u32),
            ScalarArg::new(options.stride[1] as u32),
            ScalarArg::new(options.stride[2] as u32),
            ScalarArg::new(options.dilation[0] as u32),
            ScalarArg::new(options.dilation[1] as u32),
            ScalarArg::new(options.dilation[2] as u32),
            ScalarArg::new(options.padding[0] as u32),
            ScalarArg::new(options.padding[1] as u32),
            ScalarArg::new(options.padding[2] as u32),
            ScalarArg::new(options.groups as u32),
        ),
        Some(kernel_0 as u32),
        Some(kernel_1 as u32),
        Some(kernel_2 as u32),
    );

    output
}
