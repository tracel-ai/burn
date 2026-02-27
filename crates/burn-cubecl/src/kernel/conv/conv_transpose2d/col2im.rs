use crate::{
    CubeRuntime,
    kernel::{
        conv::batches_per_run,
        into_contiguous_aligned,
        matmul::{MatmulStrategy, matmul},
        slice,
        utils::{address_type, decompose_linear, linear_view, shape_divmod},
    },
    ops::{numeric::empty_device_dtype, reshape, swap_dims},
    tensor::CubeTensor,
};
use burn_backend::{
    Shape,
    ops::{ConvTransposeOptions, conv::calculate_conv_transpose_output_size},
};
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{FastDivmod, tensor::layout::linear::LinearView},
};
use cubek::convolution::components::ConvSetupError;

/// Perform a 2D convolution transposition using the GEMM (col2im) algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv_transpose2d_col2im<R: CubeRuntime>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvTransposeOptions<2>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let [input_channels, im_ch_per_group, kernel_h, kernel_w] = weight.meta.shape().dims();
    let [batch_size, _, input_h, input_w] = input.meta.shape().dims();
    let groups = options.groups;
    let input_ch_per_group = input_channels / groups;
    let ConvTransposeOptions {
        padding: [padding_h, padding_w],
        padding_out: [padding_out_h, padding_out_w],
        dilation: [dilation_h, dilation_w],
        stride: [stride_h, stride_w],
        ..
    } = options.clone();

    let im_h = calculate_conv_transpose_output_size(
        kernel_h,
        stride_h,
        padding_h,
        padding_out_h,
        dilation_h,
        input_h,
    );
    let im_w = calculate_conv_transpose_output_size(
        kernel_w,
        stride_w,
        padding_w,
        padding_out_w,
        dilation_w,
        input_w,
    );
    let im_channels = im_ch_per_group * groups;

    let batches_per_run = batches_per_run(
        batch_size,
        input_h * input_w,
        input.client.properties().hardware.plane_size_max as usize,
    )?;
    let col_shape_0 = im_ch_per_group * kernel_h * kernel_w;

    let weight = reshape(
        weight.clone(),
        Shape::new([groups, input_ch_per_group, col_shape_0]),
    );
    let weight = into_contiguous_aligned(swap_dims(weight, 1, 2));

    if batches_per_run != batch_size {
        let runs = batch_size / batches_per_run;

        let im_shape = Shape::new([runs, batches_per_run, im_channels, im_h, im_w]);
        let image = empty_device_dtype(
            input.client.clone(),
            input.device.clone(),
            im_shape,
            input.dtype,
        );

        let input_shape = Shape::new([runs, batches_per_run, input_channels, input_h, input_w]);
        let input = reshape(input, input_shape);
        let input_shape_run = Shape::new([batches_per_run, input_channels, input_h, input_w]);

        for run in 0..runs {
            let input = index(input.clone(), run);
            let input = reshape(input, input_shape_run.clone());
            let im_shape = Shape::new([batches_per_run, im_channels, im_h, im_w]);
            let image_slice = index(image.clone(), run);
            let image_slice = reshape(image_slice, im_shape);
            execute(
                input,
                weight.clone(),
                bias.clone(),
                image_slice,
                options.clone(),
                kernel_h,
                kernel_w,
            )?;
        }
        Ok(reshape(
            image,
            Shape::new([batch_size, im_channels, im_h, im_w]),
        ))
    } else {
        let im_shape = Shape::new([batches_per_run, im_channels, im_h, im_w]);
        let image = empty_device_dtype(
            input.client.clone(),
            input.device.clone(),
            im_shape,
            input.dtype,
        );
        execute(
            input,
            weight,
            bias,
            image.clone(),
            options,
            kernel_h,
            kernel_w,
        )?;
        Ok(image)
    }
}

pub(crate) fn index<R: CubeRuntime>(tensor: CubeTensor<R>, i: usize) -> CubeTensor<R> {
    #[allow(clippy::single_range_in_vec_init)]
    let mut indices = vec![i..i + 1];
    for dim in tensor.meta.shape()[1..].iter() {
        indices.push(0..*dim);
    }
    let mut tensor = slice(tensor, &indices);
    tensor.meta.remove(0);
    tensor
}

#[allow(clippy::too_many_arguments)]
fn execute<R: CubeRuntime>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    image: CubeTensor<R>,
    options: ConvTransposeOptions<2>,
    kernel_h: usize,
    kernel_w: usize,
) -> Result<(), ConvSetupError> {
    let [batch_size, _, input_h, input_w] = input.meta.shape().dims();
    let [groups, col_shape_0, input_ch_per_group] = weight.meta.shape().dims();

    let col_shape_1 = batch_size * input_h * input_w;

    let input = swap_dims(input, 0, 1);
    let input_shape = Shape::new([groups, input_ch_per_group, col_shape_1]);
    let input = reshape(input, input_shape);

    let dtype = input.dtype;
    let columns = matmul(weight, input, None, MatmulStrategy::default(), dtype)?;
    let columns = reshape(columns, Shape::new([col_shape_0 * groups, col_shape_1]));

    col2im(
        columns, bias, image, kernel_h, kernel_w, input_h, input_w, options,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn col2im<R: CubeRuntime>(
    columns: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    out: CubeTensor<R>,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
    options: ConvTransposeOptions<2>,
) -> Result<(), LaunchError> {
    let dtype = columns.dtype;

    let columns = into_contiguous_aligned(columns);
    let bias = bias.map(into_contiguous_aligned);

    let num_elems = out.meta.num_elements();

    let cube_dim = CubeDim::new(&columns.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&columns.client, num_elems, cube_dim);

    unsafe {
        col2im_kernel::launch_unchecked(
            &columns.client,
            cube_count,
            cube_dim,
            address_type!(columns, bias, out),
            columns.as_tensor_arg(1),
            bias.as_ref().map(|bias| bias.as_tensor_arg(1)).into(),
            linear_view(&out, 1),
            shape_divmod(&out),
            Col2ImArgsLaunch::new(
                ScalarArg::new(out_h),
                ScalarArg::new(out_w),
                ScalarArg::new(kernel_h),
                ScalarArg::new(kernel_w),
                ScalarArg::new(options.padding[0]),
                ScalarArg::new(options.padding[1]),
                ScalarArg::new(options.dilation[0]),
                ScalarArg::new(options.dilation[1]),
                ScalarArg::new(options.stride[0]),
                ScalarArg::new(options.stride[1]),
            ),
            dtype.into(),
        )
    };

    Ok(())
}

#[derive(CubeLaunch, CubeType)]
struct Col2ImArgs {
    out_h: usize,
    out_w: usize,

    kernel_h: usize,
    kernel_w: usize,

    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    stride_h: usize,
    stride_w: usize,
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn col2im_kernel<E: Numeric>(
    columns: &Tensor<E>,
    bias: &Option<Tensor<E>>,
    image: &mut LinearView<E, ReadWrite>,
    image_shape: Sequence<FastDivmod<usize>>,
    args: &Col2ImArgs,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= image.shape() {
        terminate!();
    }

    let (_, pos) = decompose_linear(ABSOLUTE_POS, &image_shape);
    let [batch, ch_im, im_y, im_x] = *pos else {
        unreachable!()
    };

    let im_x = im_x + args.pad_w;
    let im_y = im_y + args.pad_h;

    let kernel_extent_w = (args.kernel_w - 1) * args.dilation_w + 1;
    let kernel_extent_h = (args.kernel_h - 1) * args.dilation_h + 1;

    let mut val = E::from_int(0);

    let x_col_start = if im_x >= kernel_extent_w {
        (im_x - kernel_extent_w) / args.stride_w + 1
    } else {
        0usize.runtime()
    };
    let x_col_end = clamp_max(im_x / args.stride_w + 1, args.out_w);
    let y_col_start = if im_y >= kernel_extent_h {
        (im_y - kernel_extent_h) / args.stride_h + 1
    } else {
        0usize.runtime()
    };
    let y_col_end = clamp_max(im_y / args.stride_h + 1, args.out_h);

    for col_y in y_col_start..y_col_end {
        let kernel_y = im_y - col_y * args.stride_h;
        for col_x in x_col_start..x_col_end {
            let kernel_x = im_x - col_x * args.stride_w;

            if kernel_y.is_multiple_of(args.dilation_h) && kernel_x.is_multiple_of(args.dilation_w)
            {
                let kernel_y = kernel_y / args.dilation_h;
                let kernel_x = kernel_x / args.dilation_w;

                let col_k =
                    ch_im * args.kernel_h * args.kernel_w + kernel_y * args.kernel_w + kernel_x;
                let col_n = batch * args.out_h * args.out_w + col_y * args.out_w + col_x;
                let col_pos = col_k * columns.stride(0) + col_n * columns.stride(1);
                val += columns[col_pos];
            }
        }
    }

    match bias {
        Some(bias) => image[ABSOLUTE_POS] = val + bias[ch_im],
        None => image[ABSOLUTE_POS] = val,
    }
}
