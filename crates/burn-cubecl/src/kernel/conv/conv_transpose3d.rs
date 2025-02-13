use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{
    element::CubeElement,
    kernel::into_contiguous,
    ops::{
        numeric::{empty_device, zeros_device},
        reshape,
    },
    tensor::CubeTensor,
    CubeRuntime,
};
use burn_tensor::{ops::ConvTransposeOptions, Element, Shape};

#[derive(CubeLaunch)]
struct ConvArgs {
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
fn conv_transpose3d_kernel<E: Numeric>(
    input: &Tensor<E>,
    weight: &Tensor<E>,
    bias: &Tensor<E>,
    output: &mut Tensor<E>,
    args: ConvArgs,
) {
    let in_channels = weight.shape(0);
    let out_c_per_group = weight.shape(1);
    let kernel_size_0 = weight.shape(2);
    let kernel_size_1 = weight.shape(3);
    let kernel_size_2 = weight.shape(4);

    let stride_0_i = args.conv_stride_0 as i32;
    let stride_1_i = args.conv_stride_1 as i32;
    let stride_2_i = args.conv_stride_2 as i32;

    let batch = ABSOLUTE_POS / output.stride(0) % output.shape(0);
    let out_c_out = ABSOLUTE_POS / output.stride(1) % output.shape(1);
    let out_z = ABSOLUTE_POS / output.stride(2) % output.shape(2);
    let out_y = ABSOLUTE_POS / output.stride(3) % output.shape(3);
    let out_x = ABSOLUTE_POS / output.stride(4) % output.shape(4);

    let groups = args.groups;
    let in_c_per_group = in_channels / groups;

    let k = out_c_out / out_c_per_group;
    let group = k % groups;
    let out_channel = out_c_out - out_c_per_group * group;

    let in_c_start = group * in_c_per_group;
    let in_c_end = in_c_start + in_c_per_group;

    let kernel_d = (kernel_size_0 * args.dilation_0 - args.conv_stride_0) as i32;
    let kernel_h = (kernel_size_1 * args.dilation_1 - args.conv_stride_1) as i32;
    let kernel_w = (kernel_size_2 * args.dilation_2 - args.conv_stride_2) as i32;

    let z_start = ((out_z + args.padding_0) as i32 - kernel_d) / stride_0_i;
    let y_start = ((out_y + args.padding_1) as i32 - kernel_h) / stride_1_i;
    let x_start = ((out_x + args.padding_2) as i32 - kernel_w) / stride_2_i;

    let z_end = Min::min(Max::max(kernel_d + z_start + 1, 0) as u32, input.shape(2));
    let y_end = Min::min(Max::max(kernel_h + y_start + 1, 0) as u32, input.shape(3));
    let x_end = Min::min(Max::max(kernel_w + x_start + 1, 0) as u32, input.shape(4));

    let z_start = Max::max(z_start, 0) as u32;
    let y_start = Max::max(y_start, 0) as u32;
    let x_start = Max::max(x_start, 0) as u32;

    let index_input_batch = batch * input.stride(0);
    let index_weight_out_c = out_channel * weight.stride(1);

    let mut sum = bias[out_c_out];

    let numerator_d_base = out_z + args.padding_0;
    let numerator_h_base = out_y + args.padding_1;
    let numerator_w_base = out_x + args.padding_2;

    for in_c in in_c_start..in_c_end {
        let index_input_in_c = in_c * input.stride(1);
        let index_weight_in_c = in_c * weight.stride(0);

        for in_z in z_start..z_end {
            let numerator_tmp = in_z * args.conv_stride_0;
            let numerator_d = numerator_d_base - numerator_tmp;

            if numerator_d_base >= numerator_tmp && numerator_d % args.dilation_0 == 0 {
                let kernel_z = numerator_d / args.dilation_0;
                let index_input_z = in_z * input.stride(2);
                let index_weight_kz = kernel_z * weight.stride(2);

                for in_y in y_start..y_end {
                    let numerator_tmp = in_y * args.conv_stride_1;
                    let numerator_h = numerator_h_base - numerator_tmp;

                    if numerator_h_base >= numerator_tmp && numerator_h % args.dilation_1 == 0 {
                        let kernel_y = numerator_h / args.dilation_1;
                        let index_input_y = in_y * input.stride(3);
                        let index_weight_ky = kernel_y * weight.stride(3);

                        for in_x in x_start..x_end {
                            let numerator_tmp = in_x * args.conv_stride_2;
                            let numerator_w = numerator_w_base - numerator_tmp;

                            if numerator_w_base >= numerator_tmp
                                && numerator_w % args.dilation_2 == 0
                            {
                                let kernel_x = numerator_w / args.dilation_2;
                                let index_input_x = in_x * input.stride(4);
                                let index_weight_kx = kernel_x * weight.stride(4);

                                let index_input = index_input_batch
                                    + index_input_in_c
                                    + index_input_z
                                    + index_input_y
                                    + index_input_x;

                                let index_weight = index_weight_in_c
                                    + index_weight_out_c
                                    + index_weight_kz
                                    + index_weight_ky
                                    + index_weight_kx;

                                let value = input[index_input];
                                let weight = weight[index_weight];

                                sum += value * weight;
                            }
                        }
                    }
                }
            }
        }
    }

    output[ABSOLUTE_POS] = sum;
}

pub(crate) fn conv_transpose3d<R: CubeRuntime, E: CubeElement + Element>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvTransposeOptions<3>,
) -> CubeTensor<R> {
    let input = into_contiguous(input);
    let weight = into_contiguous(weight);
    let [batch_size, _, in_depth, in_height, in_width] = input.shape.dims();
    let [_, out_channels, kernel_0, kernel_1, kernel_2] = weight.shape.dims();

    let out_0 = (in_depth - 1) * options.stride[0]
        + options.dilation[0] * (kernel_0 - 1)
        + options.padding_out[0]
        - 2 * options.padding[0]
        + 1;
    let out_1 = (in_height - 1) * options.stride[1]
        + options.dilation[1] * (kernel_1 - 1)
        + options.padding_out[1]
        - 2 * options.padding[1]
        + 1;
    let out_2 = (in_width - 1) * options.stride[2]
        + options.dilation[2] * (kernel_2 - 1)
        + options.padding_out[2]
        - 2 * options.padding[2]
        + 1;

    let shape_out = Shape::new([
        batch_size,
        out_channels * options.groups,
        out_0,
        out_1,
        out_2,
    ]);

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

    conv_transpose3d_kernel::launch::<E, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg::<E>(1),
        weight.as_tensor_arg::<E>(1),
        bias.as_tensor_arg::<E>(1),
        output.as_tensor_arg::<E>(1),
        ConvArgsLaunch::new(
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
    );

    output
}
