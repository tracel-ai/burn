use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{FastDivmod, tensor::layout::linear::LinearView},
};

use crate::{
    CubeRuntime,
    kernel::utils::{address_type, decompose_linear, linear_view, shape_divmod},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use burn_backend::{Shape, ops::ConvTransposeOptions};

#[derive(CubeLaunch, CubeType)]
struct ConvArgs {
    conv_stride_0: usize,
    conv_stride_1: usize,
    conv_stride_2: usize,
    dilation_0: usize,
    dilation_1: usize,
    dilation_2: usize,
    padding_0: usize,
    padding_1: usize,
    padding_2: usize,
    groups: usize,
}

#[cube(launch, address_type = "dynamic")]
fn conv_transpose3d_kernel<E: Numeric>(
    input: &Tensor<E>,
    weight: &Tensor<E>,
    bias: &Option<Tensor<E>>,
    output: &mut LinearView<E, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    args: ConvArgs,
    #[define(E)] _dtype: StorageType,
) {
    let in_channels = weight.shape(0);
    let out_c_per_group = weight.shape(1);
    let kernel_size_0 = weight.shape(2);
    let kernel_size_1 = weight.shape(3);
    let kernel_size_2 = weight.shape(4);

    let stride_0_i = args.conv_stride_0 as i32;
    let stride_1_i = args.conv_stride_1 as i32;
    let stride_2_i = args.conv_stride_2 as i32;

    let (_, pos) = decompose_linear(ABSOLUTE_POS, &out_shape);
    let [batch, out_c_out, out_z, out_y, out_x] = *pos else {
        unreachable!()
    };

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

    let z_end = clamp(kernel_d + z_start + 1, 0, input.shape(2) as i32) as usize;
    let y_end = clamp(kernel_h + y_start + 1, 0, input.shape(3) as i32) as usize;
    let x_end = clamp(kernel_w + x_start + 1, 0, input.shape(4) as i32) as usize;

    let z_start = clamp_min(z_start, 0) as usize;
    let y_start = clamp_min(y_start, 0) as usize;
    let x_start = clamp_min(x_start, 0) as usize;

    let index_input_batch = batch * input.stride(0);
    let index_weight_out_c = out_channel * weight.stride(1);

    let bias: Option<E> = bias.map(|bias| bias[out_c_out]);
    let mut sum = bias.unwrap_or_default();

    let numerator_d_base = out_z + args.padding_0;
    let numerator_h_base = out_y + args.padding_1;
    let numerator_w_base = out_x + args.padding_2;

    for in_c in in_c_start..in_c_end {
        let index_input_in_c = in_c * input.stride(1);
        let index_weight_in_c = in_c * weight.stride(0);

        for in_z in z_start..z_end {
            let numerator_tmp = in_z * args.conv_stride_0;
            let numerator_d = numerator_d_base - numerator_tmp;

            if numerator_d_base >= numerator_tmp && numerator_d.is_multiple_of(args.dilation_0) {
                let kernel_z = numerator_d / args.dilation_0;
                let index_input_z = in_z * input.stride(2);
                let index_weight_kz = kernel_z * weight.stride(2);

                for in_y in y_start..y_end {
                    let numerator_tmp = in_y * args.conv_stride_1;
                    let numerator_h = numerator_h_base - numerator_tmp;

                    if numerator_h_base >= numerator_tmp
                        && numerator_h.is_multiple_of(args.dilation_1)
                    {
                        let kernel_y = numerator_h / args.dilation_1;
                        let index_input_y = in_y * input.stride(3);
                        let index_weight_ky = kernel_y * weight.stride(3);

                        for in_x in x_start..x_end {
                            let numerator_tmp = in_x * args.conv_stride_2;
                            let numerator_w = numerator_w_base - numerator_tmp;

                            if numerator_w_base >= numerator_tmp
                                && numerator_w.is_multiple_of(args.dilation_2)
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

pub(crate) fn conv_transpose3d<R: CubeRuntime>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvTransposeOptions<3>,
) -> Result<CubeTensor<R>, LaunchError> {
    let [batch_size, _, in_depth, in_height, in_width] = input.meta.shape().dims();
    let [_, out_channels, kernel_0, kernel_1, kernel_2] = weight.meta.shape().dims();

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

    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
        input.dtype,
    );

    let num_elems = output.meta.num_elements();
    let cube_dim = CubeDim::new(&input.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&input.client, num_elems, cube_dim);

    conv_transpose3d_kernel::launch(
        &input.client,
        cube_count,
        cube_dim,
        address_type!(input, weight, bias, output),
        input.as_tensor_arg(1),
        weight.as_tensor_arg(1),
        bias.as_ref().map(|bias| bias.as_tensor_arg(1)).into(),
        linear_view(&output, 1),
        shape_divmod(&output),
        ConvArgsLaunch::new(
            ScalarArg::new(options.stride[0]),
            ScalarArg::new(options.stride[1]),
            ScalarArg::new(options.stride[2]),
            ScalarArg::new(options.dilation[0]),
            ScalarArg::new(options.dilation[1]),
            ScalarArg::new(options.dilation[2]),
            ScalarArg::new(options.padding[0]),
            ScalarArg::new(options.padding[1]),
            ScalarArg::new(options.padding[2]),
            ScalarArg::new(options.groups),
        ),
        input.dtype.into(),
    )?;

    Ok(output)
}
