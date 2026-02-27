use crate::{
    CubeRuntime,
    kernel::utils::{address_type, decompose_linear, linear_view, shape_divmod},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use burn_backend::{Shape, ops::ConvTransposeOptions};
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{FastDivmod, tensor::layout::linear::LinearView},
};
use cubek::convolution::components::ConvSetupError;

#[derive(CubeLaunch, CubeType)]
struct ConvArgs {
    conv_stride_0: usize,
    conv_stride_1: usize,
    dilation_0: usize,
    dilation_1: usize,
    padding_0: usize,
    padding_1: usize,
    groups: usize,
}

#[cube(launch, address_type = "dynamic")]
fn conv_transpose2d_direct_kernel<E: Numeric>(
    input: &Tensor<E>,
    weight: &Tensor<E>,
    bias: &Option<Tensor<E>>,
    output: &mut LinearView<E, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    args: ConvArgs,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.shape() {
        terminate!();
    }

    let in_c_per_group = weight.shape(0) / args.groups;
    let out_c_per_group = weight.shape(1);
    let kernel_h = weight.shape(2);
    let kernel_w = weight.shape(3);

    let (_, pos) = decompose_linear(ABSOLUTE_POS, &out_shape);
    let [batch, oc_out, out_y, out_x] = *pos else {
        unreachable!()
    };

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

    let y_end = clamp(kms_h + y_start + 1, 0, input.shape(2) as i32) as usize;
    let x_end = clamp(kms_w + x_start + 1, 0, input.shape(3) as i32) as usize;
    let y_start = clamp_min(y_start, 0) as usize;
    let x_start = clamp_min(x_start, 0) as usize;

    let idx_input_batch = batch * input.stride(0);
    let idx_weight_oc = out_c * weight.stride(1);

    let bias: Option<E> = bias.map(|bias| bias[oc_out]);
    let mut sum = bias.unwrap_or_default();

    let numerator_h_base = out_y + args.padding_0;
    let numerator_w_base = out_x + args.padding_1;

    for in_c in in_c_start..in_c_end {
        let idx_input_ic = in_c * input.stride(1);
        let idx_weight_ic = in_c * weight.stride(0);

        for in_y in y_start..y_end {
            let numerator_tmp = in_y * args.conv_stride_0;
            let numerator_h = numerator_h_base - numerator_tmp;

            if numerator_h_base >= numerator_tmp && numerator_h.is_multiple_of(args.dilation_0) {
                let kernel_y = numerator_h / args.dilation_0;
                let idx_input_y = in_y * input.stride(2);
                let idx_weight_ky = kernel_y * weight.stride(2);

                for in_x in x_start..x_end {
                    let numerator_tmp = in_x * args.conv_stride_1;
                    let numerator_w = numerator_w_base - numerator_tmp;

                    if numerator_w_base >= numerator_tmp
                        && numerator_w.is_multiple_of(args.dilation_1)
                    {
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
pub fn conv_transpose2d_direct<R: CubeRuntime>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvTransposeOptions<2>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let [batch_size, _, in_height, in_width] = input.meta.shape().dims();
    let [_, out_channels, kernel_0, kernel_1] = weight.meta.shape().dims();

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

    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
        input.dtype,
    );

    let num_elems = output.meta.num_elements();
    let cube_dim = CubeDim::new(&input.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&input.client, num_elems, cube_dim);
    let dtype = input.dtype;

    conv_transpose2d_direct_kernel::launch(
        &output.client,
        cube_count,
        cube_dim,
        address_type!(input, weight, bias, output),
        input.into_tensor_arg(1),
        weight.into_tensor_arg(1),
        bias.map(|bias| bias.into_tensor_arg(1)).into(),
        linear_view(output.clone(), 1),
        shape_divmod(&output),
        ConvArgsLaunch::new(
            ScalarArg::new(options.stride[0]),
            ScalarArg::new(options.stride[1]),
            ScalarArg::new(options.dilation[0]),
            ScalarArg::new(options.dilation[1]),
            ScalarArg::new(options.padding[0]),
            ScalarArg::new(options.padding[1]),
            ScalarArg::new(options.groups),
        ),
        dtype.into(),
    );

    Ok(output)
}
