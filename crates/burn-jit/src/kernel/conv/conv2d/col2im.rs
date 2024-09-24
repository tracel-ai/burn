use burn_tensor::{
    ops::{conv::calculate_conv_transpose_output_size, ConvTransposeOptions, FloatTensorOps as _},
    Shape,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{
    kernel::into_contiguous,
    ops::{numeric::empty_device, reshape, swap_dims},
    tensor::JitTensor,
    FloatElement, IntElement, JitBackend, JitRuntime,
};

use super::batches_per_run;

/// Perform a 2D convolution transposition using the GEMM (col2im) algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
pub fn conv_transpose2d_col2im<R: JitRuntime, E: FloatElement, I: IntElement>(
    input: JitTensor<R, E>,
    weight: JitTensor<R, E>,
    bias: Option<JitTensor<R, E>>,
    options: ConvTransposeOptions<2>,
) -> JitTensor<R, E> {
    let [input_channels, im_ch_per_group, kernel_h, kernel_w] = weight.shape.dims();
    let [batch_size, _, input_h, input_w] = input.shape.dims();
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

    let batches_per_run = batches_per_run(batch_size, input_h, input_w);
    let col_shape_0 = im_ch_per_group * kernel_h * kernel_w;

    let weight = reshape(
        weight.clone(),
        Shape::new([groups, input_ch_per_group, col_shape_0]),
    );
    let weight = into_contiguous(swap_dims(weight, 1, 2));

    if batches_per_run != batch_size {
        let runs = batch_size / batches_per_run;

        let im_shape = Shape::new([runs, batches_per_run, im_channels, im_h, im_w]);
        let mut image = empty_device(input.client.clone(), input.device.clone(), im_shape);

        let input_shape = Shape::new([runs, batches_per_run, input_channels, input_h, input_w]);
        let input = reshape(input, input_shape);
        let input_shape_run = Shape::new([batches_per_run, input_channels, input_h, input_w]);
        let im_shape_run = Shape::new([1, batches_per_run, im_channels, im_h, im_w]);

        for run in 0..runs {
            let input = JitBackend::<R, E, I>::float_narrow(input.clone(), 0, run, 1);
            let input = reshape(input, input_shape_run.clone());
            let image_run = execute::<R, E, I>(
                input,
                weight.clone(),
                bias.clone(),
                options.clone(),
                im_ch_per_group,
                im_h,
                im_w,
                kernel_h,
                kernel_w,
            );
            let image_run = reshape(image_run, im_shape_run.clone());
            image = JitBackend::<R, E, I>::float_slice_assign(
                image,
                &[
                    run..run + 1,
                    0..batches_per_run,
                    0..im_channels,
                    0..im_h,
                    0..im_w,
                ],
                image_run,
            )
        }
        reshape(image, Shape::new([batch_size, im_channels, im_h, im_w]))
    } else {
        execute::<R, E, I>(
            input,
            weight,
            bias,
            options,
            im_ch_per_group,
            im_h,
            im_w,
            kernel_h,
            kernel_w,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn execute<R: JitRuntime, E: FloatElement, I: IntElement>(
    input: JitTensor<R, E>,
    weight: JitTensor<R, E>,
    bias: Option<JitTensor<R, E>>,
    options: ConvTransposeOptions<2>,
    im_ch_per_group: usize,
    im_h: usize,
    im_w: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> JitTensor<R, E> {
    let [batch_size, _, input_h, input_w] = input.shape.dims();
    let [groups, col_shape_0, input_ch_per_group] = weight.shape.dims();

    let im_channels = im_ch_per_group * groups;

    let im_shape = Shape::new([batch_size, im_channels, im_h, im_w]);

    let col_shape_1 = batch_size * input_h * input_w;

    let input = swap_dims(input, 0, 1);
    let input_shape = Shape::new([groups, input_ch_per_group, col_shape_1]);
    let input = reshape(input, input_shape);

    let columns = JitBackend::<R, E, I>::float_matmul(weight, input);
    let columns = reshape(columns, Shape::new([col_shape_0 * groups, col_shape_1]));

    col2im(
        columns, bias, im_shape, kernel_h, kernel_w, input_h, input_w, options,
    )
}

#[allow(clippy::too_many_arguments)]
fn col2im<R: JitRuntime, E: FloatElement>(
    columns: JitTensor<R, E>,
    bias: Option<JitTensor<R, E>>,
    im_shape: Shape,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
    options: ConvTransposeOptions<2>,
) -> JitTensor<R, E> {
    let [_, col_size_1] = columns.shape.dims();

    let columns = into_contiguous(columns);
    let has_bias = bias.is_some();
    let bias = bias.map(into_contiguous).unwrap_or_else(|| {
        empty_device(
            columns.client.clone(),
            columns.device.clone(),
            Shape::new([1]),
        )
    });

    let num_elems = im_shape.num_elements();
    let out = empty_device(
        columns.client.clone(),
        columns.device.clone(),
        im_shape.clone(),
    );

    let vectorization = 1;
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);

    unsafe {
        col2im_kernel::launch_unchecked::<E, R>(
            &columns.client,
            cube_count,
            cube_dim,
            columns.as_tensor_arg(vectorization),
            bias.as_tensor_arg(vectorization),
            out.as_tensor_arg(vectorization),
            Col2ImArgsLaunch::new(
                ScalarArg::new(out_h as u32),
                ScalarArg::new(out_w as u32),
                ScalarArg::new(kernel_h as u32),
                ScalarArg::new(kernel_w as u32),
                ScalarArg::new(options.padding[0] as u32),
                ScalarArg::new(options.padding[1] as u32),
                ScalarArg::new(options.dilation[0] as u32),
                ScalarArg::new(options.dilation[1] as u32),
                ScalarArg::new(options.stride[0] as u32),
                ScalarArg::new(options.stride[1] as u32),
                ScalarArg::new(col_size_1 as u32),
            ),
            has_bias,
        )
    };

    out
}

#[derive(CubeLaunch)]
struct Col2ImArgs {
    out_h: u32,
    out_w: u32,

    kernel_h: u32,
    kernel_w: u32,

    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    stride_h: u32,
    stride_w: u32,

    col_size_1: u32,
}

#[cube(launch_unchecked)]
fn col2im_kernel<F: Float>(
    columns: &Tensor<F>,
    bias: &Tensor<F>,
    image: &mut Tensor<F>,
    args: &Col2ImArgs,
    #[comptime] has_bias: bool,
) {
    if ABSOLUTE_POS > image.len() {
        return;
    }

    let _ = bias[0]; // Keep in bind group

    let im_x = ABSOLUTE_POS % image.shape(3) + args.pad_w;
    let im_y = ABSOLUTE_POS / image.stride(2) % image.shape(2) + args.pad_h;
    let ch_im = ABSOLUTE_POS / image.stride(1) % image.shape(1);
    let batch = ABSOLUTE_POS / image.stride(0);

    let kernel_extent_w = (args.kernel_w - 1) * args.dilation_w + 1;
    let kernel_extent_h = (args.kernel_h - 1) * args.dilation_h + 1;

    let mut val = F::new(0.0);

    let x_col_start = if im_x >= kernel_extent_w {
        (im_x - kernel_extent_w) / args.stride_w + 1
    } else {
        0u32
    };
    let x_col_end = Min::min(im_x / args.stride_w + 1, args.out_w);
    let y_col_start = if im_y >= kernel_extent_h {
        (im_y - kernel_extent_h) / args.stride_h + 1
    } else {
        0u32
    };
    let y_col_end = Min::min(im_y / args.stride_h + 1, args.out_h);

    for col_y in y_col_start..y_col_end {
        let kernel_y = im_y - col_y * args.stride_h;
        for col_x in x_col_start..x_col_end {
            let kernel_x = im_x - col_x * args.stride_w;

            if kernel_y % args.dilation_h == 0 && kernel_x % args.dilation_w == 0 {
                let kernel_y = kernel_y / args.dilation_h;
                let kernel_x = kernel_x / args.dilation_w;

                let col_pos = ch_im * args.kernel_h * args.kernel_w * args.col_size_1
                    + kernel_y * args.kernel_w * args.col_size_1
                    + kernel_x * args.col_size_1
                    + batch * args.out_h * args.out_w
                    + col_y * args.out_w
                    + col_x;
                val += columns[col_pos];
            }
        }
    }

    if has_bias {
        image[ABSOLUTE_POS] = val + bias[ch_im];
    } else {
        image[ABSOLUTE_POS] = val;
    }
}
