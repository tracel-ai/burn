use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{
    element::JitElement,
    kernel::{conv::ConvLaunchError, into_contiguous},
    ops::{
        numeric::{empty_device, zeros_device},
        reshape,
    },
    tensor::JitTensor,
    JitRuntime,
};
use burn_tensor::{ops::ConvTransposeOptions, Shape};

#[derive(CubeLaunch)]
struct ConvArgs {
    conv_stride_0: u32,
    conv_stride_1: u32,
    dilation_0: u32,
    dilation_1: u32,
    padding_0: u32,
    padding_1: u32,
    groups: u32,
}

#[cube(launch)]
fn conv_transpose2d_direct_kernel<E: Numeric>(
    input: &Tensor<E>,
    weight: &Tensor<E>,
    bias: &Tensor<E>,
    output: &mut Tensor<E>,
    args: ConvArgs,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let in_c_per_group = weight.shape(0) / args.groups;
    let out_c_per_group = weight.shape(1);
    let kernel_h = weight.shape(2);
    let kernel_w = weight.shape(3);

    let batch = ABSOLUTE_POS / output.stride(0) % output.shape(0);
    let oc_out = ABSOLUTE_POS / output.stride(1) % output.shape(1);
    let out_y = ABSOLUTE_POS / output.stride(2) % output.shape(2);
    let out_x = ABSOLUTE_POS / output.stride(3) % output.shape(3);

    let k = oc_out / out_c_per_group;
    let group = k % args.groups;
    let out_c = oc_out - out_c_per_group * group;

    let in_c_start = group * in_c_per_group;
    let in_c_end = in_c_start + in_c_per_group;

    let stride_0_i = args.conv_stride_0 as i32;
    let stride_1_i = args.conv_stride_1 as i32;

    let kms_h = (kernel_h * args.dilation_0) as i32 - stride_0_i;
    let kms_w = (kernel_w * args.dilation_1) as i32 - stride_1_i;

    let y_start = ((out_y + args.padding_0) as i32 - kms_h) / stride_0_i;
    let x_start = ((out_x + args.padding_1) as i32 - kms_w) / stride_1_i;

    let y_end = Min::min(Max::max(kms_h + y_start + 1, 0) as u32, input.shape(2));
    let x_end = Min::min(Max::max(kms_w + x_start + 1, 0) as u32, input.shape(3));
    let y_start = Max::max(y_start, 0) as u32;
    let x_start = Max::max(x_start, 0) as u32;

    let idx_input_batch = batch * input.stride(0);
    let idx_weight_oc = out_c * weight.stride(1);

    let mut sum = bias[oc_out];

    let numerator_h_base = out_y + args.padding_0;
    let numerator_w_base = out_x + args.padding_1;

    for in_c in in_c_start..in_c_end {
        let idx_input_ic = in_c * input.stride(1);
        let idx_weight_ic = in_c * weight.stride(0);

        for in_y in y_start..y_end {
            let numerator_tmp = in_y * args.conv_stride_0;
            let numerator_h = numerator_h_base - numerator_tmp;

            if numerator_h_base >= numerator_tmp && numerator_h % args.dilation_0 == 0 {
                let kernel_y = numerator_h / args.dilation_0;
                let idx_input_y = in_y * input.stride(2);
                let idx_weight_ky = kernel_y * weight.stride(2);

                for in_x in x_start..x_end {
                    let numerator_tmp = in_x * args.conv_stride_1;
                    let numerator_w = numerator_w_base - numerator_tmp;

                    if numerator_w_base >= numerator_tmp && numerator_w % args.dilation_1 == 0 {
                        let kernel_x = numerator_w / args.dilation_1;
                        let idx_input_x = in_x * input.stride(3);
                        let idx_weight_kx = kernel_x * weight.stride(3);

                        let index_input =
                            idx_input_batch + idx_input_ic + idx_input_y + idx_input_x;
                        let index_weight =
                            idx_weight_ic + idx_weight_oc + idx_weight_ky + idx_weight_kx;

                        let value = input[index_input];
                        let weight = weight[index_weight];

                        sum += value * weight;
                    }
                }
            }
        }
    }

    output[ABSOLUTE_POS] = sum;
}

/// Perform a 2D convolution transposition using the direct algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
pub fn conv_transpose2d_direct<R: JitRuntime, E: JitElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    options: ConvTransposeOptions<2>,
) -> Result<JitTensor<R>, ConvLaunchError> {
    let input = into_contiguous(input);
    let weight = into_contiguous(weight);
    let [batch_size, _, in_height, in_width] = input.shape.dims();
    let [_, out_channels, kernel_0, kernel_1] = weight.shape.dims();

    let out_0 = (in_height - 1) * options.stride[0]
        + options.dilation[0] * (kernel_0 - 1)
        + options.padding_out[0]
        - 2 * options.padding[0]
        + 1;
    let out_1 = (in_width - 1) * options.stride[1]
        + options.dilation[1] * (kernel_1 - 1)
        + options.padding_out[1]
        - 2 * options.padding[1]
        + 1;

    let shape_out = Shape::new([batch_size, out_channels * options.groups, out_0, out_1]);

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

    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(output.shape.num_elements(), cube_dim);

    conv_transpose2d_direct_kernel::launch::<E, R>(
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
            ScalarArg::new(options.dilation[0] as u32),
            ScalarArg::new(options.dilation[1] as u32),
            ScalarArg::new(options.padding[0] as u32),
            ScalarArg::new(options.padding[1] as u32),
            ScalarArg::new(options.groups as u32),
        ),
    );

    Ok(output)
}
