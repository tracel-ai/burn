use crate::{
    CubeRuntime,
    kernel::{
        into_contiguous_aligned,
        utils::{address_type, linear_view},
    },
    ops::max_line_size,
    tensor::CubeTensor,
};
use crate::{kernel::utils::decompose_linear, ops::numeric::empty_device_dtype};
use burn_backend::{
    TensorMetadata,
    ops::{ConvOptions, conv::calculate_conv_output_sizes},
};
use cubecl::std::{FastDivmod, FastDivmodArgs};
use cubecl::{
    calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView,
    tensor_line_size_parallel,
};
use cubek::convolution::components::ConvSetupError;

#[derive(CubeLaunch, CubeType, Clone)]
pub(crate) struct ConvParam {
    pub stride: u32,
    pub dilation: u32,
    pub padding: i32,
}

#[derive(CubeLaunch, CubeType)]
struct Conv2dArgs {
    conv_params: Sequence<ConvParam>,
    channels_per_group: u32,
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn direct_conv2d_kernel<E: Numeric>(
    input: &Tensor<Line<E>>,
    weight: &Tensor<Line<E>>,
    bias: Option<Tensor<Line<E>>>,
    output: &mut LinearView<Line<E>, ReadWrite>,
    args: Conv2dArgs,
    shape_out: Sequence<FastDivmod<u32>>,
    shape_out_c: FastDivmod<u32>,
    #[comptime] has_padding: bool,
    #[define(E)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let n_spatial = comptime![shape_out.len()];

    let line_size_out = output.line_size();
    let pos = ABSOLUTE_POS * line_size_out;

    let in_c_per_group = weight.shape(weight.rank() - 1) as u32;

    let (rem, out_c) = shape_out_c.div_mod(pos as u32);
    let (b, spatial_pos) = decompose_linear(rem, &shape_out);

    let g = out_c / args.channels_per_group;
    let ic_start = in_c_per_group * g;

    let bias: Option<Line<E>> = bias.map(|bias| bias[out_c as usize / line_size_out]);
    let mut sum = bias.unwrap_or_else(|| Line::empty(line_size_out).fill(E::from_int(0)));

    let in_offs = b as usize * input.stride(0) + ic_start as usize;

    let stride_oc = weight.stride(0);

    let mut in_shape = Sequence::new();
    let mut in_strides = Sequence::new();
    let mut kernel_shape = Sequence::new();
    let mut kernel_strides = Sequence::new();

    #[unroll]
    for i in 0..n_spatial {
        in_shape.push(input.shape(i + 1) as u32);
        in_strides.push(input.stride(i + 1));
        kernel_shape.push(weight.shape(i + 1) as u32);
        kernel_strides.push(weight.stride(i + 1));
    }

    let weight_offs = out_c as usize * stride_oc;

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
        0usize,
        has_padding,
    );

    output[ABSOLUTE_POS] = sum;
}

#[derive(CubeType, Clone)]
struct LoopParams {
    out_pos: Sequence<u32>,
    in_shape: Sequence<u32>,
    in_strides: Sequence<usize>,
    kernel_shape: Sequence<u32>,
    kernel_strides: Sequence<usize>,
    conv_params: Sequence<ConvParam>,

    in_c_per_group: u32,
    stride_oc: usize,
}

#[cube]
fn kernel_loop<E: Numeric>(
    input: &Tensor<Line<E>>,
    weight: &Tensor<Line<E>>,
    sum: &mut Line<E>,
    in_offs: usize,
    in_bounds: bool,
    weight_offs: usize,
    params: &LoopParams,
    #[comptime] kernel_dim: usize,
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
            let in_offs = in_offs + in_pos as usize * stride;
            let weight_offs = weight_offs + pos as usize * k_stride;
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
    in_offs: usize,
    in_bounds: bool,
    weight_offs: usize,
    in_c_per_group: u32,
    stride_oc: usize,
) {
    let line_size_in = input.line_size();
    let line_size_out = sum.size();

    if in_bounds {
        for in_c in range_stepped(0, in_c_per_group, line_size_in as u32) {
            let in_pos = in_offs + in_c as usize;
            let mut weight_pos = weight_offs + in_c as usize;

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
pub fn conv_direct<R: CubeRuntime, const N: usize>(
    mut input: CubeTensor<R>,
    mut weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let client = input.client.clone();
    let out_dtype = input.dtype;
    let rank = input.meta.shape().num_dims();
    let dim_c = rank - 1;

    // We only care about the channels here, everything else can be permuted
    if input.meta.strides()[dim_c] != 1 {
        input = into_contiguous_aligned(input);
    }
    if weight.meta.strides()[dim_c] != 1 {
        weight = into_contiguous_aligned(weight);
    }

    let batch_size = input.meta.shape()[0];
    let in_shape = &input.meta.shape()[1..dim_c];
    let out_channels = weight.meta.shape()[0];
    let kernel_shape = &weight.meta.shape()[1..dim_c];

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

    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape_out.into(),
        out_dtype,
    );

    // Need custom line size calculation here to account for the groups division. Need to vectorize
    // over `channels_per_group` instead.
    let mut grouped_out_shape = output.shape();
    grouped_out_shape[dim_c] = channels_per_group;
    let line_size_out = tensor_line_size_parallel(
        input.client.io_optimized_line_sizes(input.dtype.size()),
        &grouped_out_shape,
        output.meta.strides(),
        dim_c,
    );
    // Use channels_per_group instead of in_channels to avoid issues here
    let line_size_in = max_line_size(&weight);

    let shape_out = output.meta.shape()[1..dim_c]
        .iter()
        .map(|s| FastDivmodArgs::<u32>::new(&client, *s as u32))
        .collect();
    let shape_out_c = FastDivmodArgs::<u32>::new(&client, out_channels as u32);

    let mut conv_params = SequenceArg::new();

    for i in 0..kernel_shape.len() {
        conv_params.push(ConvParamLaunch::new(
            ScalarArg::new(options.stride[i] as u32),
            ScalarArg::new(options.dilation[i] as u32),
            ScalarArg::new(options.padding[i] as i32),
        ));
    }

    let working_units = output.meta.num_elements() / line_size_out;
    let cube_dim = CubeDim::new(&input.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&input.client, working_units, cube_dim);

    unsafe {
        direct_conv2d_kernel::launch_unchecked(
            &input.client,
            cube_count,
            cube_dim,
            address_type!(input, weight, bias, output),
            input.as_tensor_arg(line_size_in),
            weight.as_tensor_arg(line_size_in),
            bias.as_ref().map(|b| b.as_tensor_arg(line_size_out)).into(),
            linear_view(&output, line_size_out),
            Conv2dArgsLaunch::new(conv_params, ScalarArg::new(channels_per_group as u32)),
            shape_out,
            shape_out_c,
            options.padding.iter().any(|it| *it != 0),
            out_dtype.into(),
        )
    };

    Ok(output)
}
