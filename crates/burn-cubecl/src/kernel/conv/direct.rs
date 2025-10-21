use burn_tensor::ops::{ConvOptions, conv::calculate_conv_output_sizes};
use cubecl::{
    calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView,
    tensor_line_size_parallel,
};
use cubecl::{
    convolution::components::ConvSetupError,
    std::{CubeOption, CubeOptionExpand, FastDivmod},
};

use crate::{
    CubeElement, CubeRuntime,
    kernel::{
        conv::div_mod_seq,
        into_contiguous_aligned,
        utils::{linear_view, shape_divmod},
    },
    ops::{max_line_size, numeric::empty_device_optimized},
    tensor::CubeTensor,
};

use super::im2col::{ConvParam, ConvParamLaunch};

#[derive(CubeLaunch, CubeType)]
struct Conv2dArgs {
    conv_params: Sequence<ConvParam>,
    channels_per_group: u32,
}

#[cube(launch_unchecked)]
fn direct_conv2d_kernel<E: Numeric>(
    input: &Tensor<Line<E>>,
    weight: &Tensor<Line<E>>,
    bias: CubeOption<Tensor<Line<E>>>,
    output: &mut LinearView<Line<E>, ReadWrite>,
    args: Conv2dArgs,
    shape_out: Sequence<FastDivmod>,
    shape_out_c: FastDivmod,
    #[comptime] has_padding: bool,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let n_spatial = comptime![shape_out.len()];

    let line_size_out = output.line_size();
    let pos = ABSOLUTE_POS * line_size_out;

    let in_c_per_group = weight.shape(weight.rank() - 1);

    let (rem, out_c) = shape_out_c.div_mod(pos);
    let (b, spatial_pos) = div_mod_seq(rem, &shape_out);

    let g = out_c / args.channels_per_group;
    let ic_start = in_c_per_group * g;

    let mut sum = match bias {
        CubeOption::Some(bias) => bias[out_c / line_size_out],
        CubeOption::None => Line::empty(line_size_out).fill(E::from_int(0)),
    };

    let in_offs = b * input.stride(0) + ic_start;

    let stride_oc = weight.stride(0);

    let mut in_shape = Sequence::new();
    let mut in_strides = Sequence::new();
    let mut kernel_shape = Sequence::new();
    let mut kernel_strides = Sequence::new();

    #[unroll]
    for i in 0..n_spatial {
        in_shape.push(input.shape(i + 1));
        in_strides.push(input.stride(i + 1));
        kernel_shape.push(weight.shape(i + 1));
        kernel_strides.push(weight.stride(i + 1));
    }

    let weight_offs = out_c * stride_oc;

    let loop_params = LoopParams {
        out_pos: spatial_pos,
        in_shape,
        in_strides,
        kernel_shape,
        kernel_strides,
        conv_params: args.conv_params,
        in_c_per_group,
        stride_oc,
    };

    kernel_loop(
        input,
        weight,
        &mut sum,
        in_offs,
        true,
        weight_offs,
        &loop_params,
        0u32,
        has_padding,
    );

    output[ABSOLUTE_POS] = sum;
}

#[derive(CubeType, Clone)]
struct LoopParams {
    out_pos: Sequence<u32>,
    in_shape: Sequence<u32>,
    in_strides: Sequence<u32>,
    kernel_shape: Sequence<u32>,
    kernel_strides: Sequence<u32>,
    conv_params: Sequence<ConvParam>,

    in_c_per_group: u32,
    stride_oc: u32,
}

#[cube]
fn kernel_loop<E: Numeric>(
    input: &Tensor<Line<E>>,
    weight: &Tensor<Line<E>>,
    sum: &mut Line<E>,
    in_offs: u32,
    in_bounds: bool,
    weight_offs: u32,
    params: &LoopParams,
    #[comptime] kernel_dim: u32,
    #[comptime] has_padding: bool,
) {
    if comptime![kernel_dim < params.kernel_shape.len()] {
        let out_idx = *params.out_pos.index(kernel_dim);
        let conv = params.conv_params.index(kernel_dim);
        let shape = *params.in_shape.index(kernel_dim);
        let stride = *params.in_strides.index(kernel_dim);
        let k_stride = *params.kernel_strides.index(kernel_dim);

        for pos in 0..*params.kernel_shape.index(kernel_dim) {
            let in_pos = (out_idx * conv.stride + pos * conv.dilation) as i32 - conv.padding;
            let in_offs = in_offs + in_pos as u32 * stride;
            let weight_offs = weight_offs + pos * k_stride;
            let mut in_bounds = in_bounds;

            if has_padding {
                in_bounds &= in_pos >= 0 && (in_pos as u32) < shape;
            }

            kernel_loop(
                input,
                weight,
                sum,
                in_offs,
                in_bounds,
                weight_offs,
                params,
                comptime![kernel_dim + 1],
                has_padding,
            );
        }
    } else {
        kernel_loop_inner(
            input,
            weight,
            sum,
            in_offs,
            in_bounds,
            weight_offs,
            params.in_c_per_group,
            params.stride_oc,
        );
    }
}

#[cube]
fn kernel_loop_inner<E: Numeric>(
    input: &Tensor<Line<E>>,
    weight: &Tensor<Line<E>>,
    sum: &mut Line<E>,
    in_offs: u32,
    in_bounds: bool,
    weight_offs: u32,
    in_c_per_group: u32,
    stride_oc: u32,
) {
    let line_size_in = input.line_size();
    let line_size_out = sum.size();

    if in_bounds {
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

/// Perform a 2D convolution using the direct convolution algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
pub fn conv_direct<R: CubeRuntime, E: CubeElement, const N: usize>(
    mut input: CubeTensor<R>,
    mut weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let rank = input.shape.num_dims();
    let dim_c = rank - 1;

    // We only care about the channels here, everything else can be permuted
    if input.strides[dim_c] != 1 {
        input = into_contiguous_aligned(input);
    }
    if weight.strides[dim_c] != 1 {
        weight = into_contiguous_aligned(weight);
    }

    let batch_size = input.shape[0];
    let in_shape = &input.shape[1..dim_c];
    let out_channels = weight.shape[0];
    let kernel_shape = &weight.shape[1..dim_c];

    let channels_per_group = out_channels / options.groups;

    let out_size = calculate_conv_output_sizes(
        kernel_shape,
        &options.stride,
        &options.padding,
        &options.dilation,
        in_shape,
    );

    let mut shape_out = vec![batch_size];
    shape_out.extend(out_size.iter().copied());
    shape_out.push(out_channels);

    let output = empty_device_optimized::<R, E>(
        input.client.clone(),
        input.device.clone(),
        shape_out.into(),
    );

    // Need custom line size calculation here to account for the groups division. Need to vectorize
    // over `channels_per_group` instead.
    let mut grouped_out_shape = output.shape.clone();
    grouped_out_shape[dim_c] = channels_per_group;
    let line_size_out = tensor_line_size_parallel(
        R::supported_line_sizes().iter().copied(),
        &grouped_out_shape,
        &output.strides,
        dim_c,
    );
    // Use channels_per_group instead of in_channels to avoid issues here
    let line_size_in = max_line_size(&weight);

    let mut shape_out = shape_divmod(&output);
    shape_out.values.remove(0);
    let shape_out_c = shape_out.values.pop().unwrap();

    let mut conv_params = SequenceArg::new();

    for i in 0..kernel_shape.len() {
        conv_params.push(ConvParamLaunch::new(
            ScalarArg::new(options.stride[i] as u32),
            ScalarArg::new(options.dilation[i] as u32),
            ScalarArg::new(options.padding[i] as i32),
        ));
    }

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
            linear_view(&output, line_size_out),
            Conv2dArgsLaunch::new(conv_params, ScalarArg::new(channels_per_group as u32)),
            shape_out,
            shape_out_c,
            options.padding.iter().any(|it| *it != 0),
        )
    };

    Ok(output)
}
