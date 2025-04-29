use burn_tensor::{
    Shape,
    ops::{ConvOptions, conv::calculate_conv_output_size},
};
use cubecl::{
    calculate_cube_count_elemwise,
    linalg::{convolution::ConvLaunchError, tensor::StridedLayout},
    prelude::*,
    tensor_line_size_parallel,
};
use cubecl_std::{CubeOption, CubeOptionExpand, FastDivmod};

use crate::{
    CubeRuntime, FloatElement,
    kernel::{
        into_contiguous_aligned,
        utils::{shape_divmod, strided_layout},
    },
    ops::{max_line_size, numeric::empty_device_strided},
    tensor::CubeTensor,
};

#[derive(CubeLaunch, CubeType)]
struct Conv2dArgs {
    stride_y: u32,
    stride_x: u32,
    dilation_y: u32,
    dilation_x: u32,
    padding_y: i32,
    padding_x: i32,
    channels_per_group: u32,
}

#[cube(launch_unchecked)]
fn direct_conv2d_kernel<E: Numeric>(
    input: &Tensor<Line<E>>,
    weight: &Tensor<Line<E>>,
    bias: CubeOption<Tensor<Line<E>>>,
    output: &mut Tensor<Line<E>>,
    args: &Conv2dArgs,
    shape_out: Sequence<FastDivmod>,
    layout_out: StridedLayout,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let out_pos = layout_out.index(output, ABSOLUTE_POS);

    let line_size_in = input.line_size();
    let line_size_out = output.line_size();
    let pos = ABSOLUTE_POS * line_size_out;

    let in_h = input.shape(1) as i32;
    let in_w = input.shape(2) as i32;
    let in_c_per_group = weight.shape(3);

    let kernel_h = weight.shape(1);
    let kernel_w = weight.shape(2);

    let (rem, out_c) = shape_out.index(3).div_mod(pos);
    let (rem, out_x) = shape_out.index(2).div_mod(rem);
    let (b, out_y) = shape_out.index(1).div_mod(rem);

    let g = out_c / args.channels_per_group;
    let ic_start = in_c_per_group * g;

    let mut sum = match bias {
        CubeOption::Some(bias) => bias[out_c / line_size_out],
        CubeOption::None => Line::empty(line_size_out).fill(E::from_int(0)),
    };

    let in_offs = b * input.stride(0) + ic_start;

    let stride_y = input.stride(1);
    let stride_x = input.stride(2);

    let stride_oc = weight.stride(0);
    let stride_kh = weight.stride(1);
    let stride_kw = weight.stride(2);

    let weight_offs = out_c * stride_oc;

    for kernel_y in 0..kernel_h {
        let in_y = (out_y * args.stride_y + kernel_y * args.dilation_y) as i32 - args.padding_y;
        let in_offs = in_offs + in_y as u32 * stride_y;
        let weight_offs = weight_offs + kernel_y * stride_kh;

        for kernel_x in 0..kernel_w {
            let in_x = (out_x * args.stride_x + kernel_x * args.dilation_x) as i32 - args.padding_x;

            if in_y >= 0 && in_y < in_h && in_x >= 0 && in_x < in_w {
                let in_offs = in_offs + in_x as u32 * stride_x;
                let weight_offs = weight_offs + kernel_x * stride_kw;

                for in_c in range_stepped(0, in_c_per_group, line_size_in) {
                    let in_pos = in_offs + in_c;
                    let mut weight_pos = weight_offs + in_c;

                    let val = input[in_pos / line_size_in];

                    #[unroll]
                    for v in 0..line_size_out {
                        let weight = weight[weight_pos / line_size_in];
                        let val = val * weight;

                        #[unroll]
                        for i in 0..line_size_in {
                            sum[v] += val[i];
                        }
                        weight_pos += stride_oc;
                    }
                }
            }
        }
    }

    output[out_pos] = sum;
}

/// Perform a 2D convolution using the direct convolution algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
pub fn conv2d_direct<R: CubeRuntime, E: FloatElement>(
    mut input: CubeTensor<R>,
    mut weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<2>,
) -> Result<CubeTensor<R>, ConvLaunchError> {
    let [batch_size, in_height, in_width, _] = input.shape.dims();
    let [out_channels, kernel_h, kernel_w, _] = weight.shape.dims();
    let channels_per_group = out_channels / options.groups;

    // We only care about the channels here, everything else can be permuted
    if input.strides[3] != 1 {
        input = into_contiguous_aligned(input);
    }
    if weight.strides[3] != 1 {
        weight = into_contiguous_aligned(weight);
    }

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

    let shape_out = Shape::new([batch_size, out_h, out_w, out_channels]);
    let output =
        empty_device_strided::<R, E>(input.client.clone(), input.device.clone(), shape_out);

    // Need custom line size calculation here to account for the groups division. Need to vectorize
    // over `channels_per_group` instead.
    let mut grouped_out_shape = output.shape.dims.clone();
    grouped_out_shape[3] = channels_per_group;
    let line_size_out = tensor_line_size_parallel(
        R::supported_line_sizes().iter().copied(),
        &grouped_out_shape,
        &output.strides,
        3,
    );
    // Use channels_per_group instead of in_channels to avoid issues here
    let line_size_in = max_line_size(&weight);

    let shape_out = shape_divmod(&output);
    let layout_out = strided_layout(&output);
    let bias = bias.as_ref().map(|b| b.as_tensor_arg::<E>(line_size_out));

    let num_elems_output = output.shape.num_elements() / line_size_out as usize;
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems_output, cube_dim);

    unsafe {
        direct_conv2d_kernel::launch_unchecked::<E, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_tensor_arg::<E>(line_size_in),
            weight.as_tensor_arg::<E>(line_size_in),
            bias.into(),
            output.as_tensor_arg::<E>(line_size_out),
            Conv2dArgsLaunch::new(
                ScalarArg::new(options.stride[0] as u32),
                ScalarArg::new(options.stride[1] as u32),
                ScalarArg::new(options.dilation[0] as u32),
                ScalarArg::new(options.dilation[1] as u32),
                ScalarArg::new(options.padding[0] as i32),
                ScalarArg::new(options.padding[1] as i32),
                ScalarArg::new(channels_per_group as u32),
            ),
            shape_out,
            layout_out,
        )
    };

    Ok(output)
}
