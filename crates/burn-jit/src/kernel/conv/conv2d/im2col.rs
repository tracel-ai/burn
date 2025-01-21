use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    Shape,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{
    kernel::{
        conv::{index, ConvLaunchError},
        into_contiguous, launch_binop,
        matmul::{matmul, MatmulStrategy},
        AddOp,
    },
    ops::{numeric::empty_device, reshape, swap_dims},
    tensor::JitTensor,
    FloatElement, JitRuntime,
};

#[derive(CubeLaunch)]
struct Im2ColArgs {
    stride_h: u32,
    stride_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    padding_h: u32,
    padding_w: u32,

    kernel_h: u32,
    kernel_w: u32,
    out_h: u32,
    out_w: u32,

    col_size_1: u32,
    num_elements: u32,
}

#[cube(launch_unchecked)]
fn im2col_kernel<F: Float>(
    image: &Tensor<F>,
    columns: &mut Tensor<F>,
    args: &Im2ColArgs,
    #[comptime] kernel_w_unroll: Option<u32>,
    #[comptime] has_padding: bool,
) {
    // position shape: [in_channels, batch_size, out_h, out_w]
    // columns shape: [[in_channels, kernel_h, kernel_w], [batch_size, out_h, out_w]]

    let batch_size = image.shape(0);
    let height = image.shape(2);
    let width = image.shape(3);

    let out_h = args.out_h;
    let out_w = args.out_w;

    if ABSOLUTE_POS > args.num_elements {
        return;
    }

    let out_x = ABSOLUTE_POS % out_w;
    let out_y = ABSOLUTE_POS / out_w % out_h;
    let batch = ABSOLUTE_POS / (out_w * out_h) % batch_size;
    let channel = ABSOLUTE_POS / (out_w * out_h * batch_size) % image.shape(1);

    let kernel_w = kernel_w_unroll.unwrap_or(args.kernel_w);
    let unroll_w = kernel_w_unroll.is_some();

    let image_idx = batch * image.stride(0) + channel * image.stride(1);
    let col_idx = channel * args.kernel_h * kernel_w * args.col_size_1
        + batch * out_h * out_w
        + out_y * out_w
        + out_x;

    for kernel_y in 0..args.kernel_h {
        #[unroll(unroll_w)]
        for kernel_x in 0..kernel_w {
            let kernel_pos = kernel_y * kernel_w + kernel_x;
            let col_pos = col_idx + kernel_pos * args.col_size_1;

            if has_padding {
                let y = (out_y * args.stride_h + kernel_y * args.dilation_h) as i32
                    - args.padding_h as i32;
                let x = (out_x * args.stride_w + kernel_x * args.dilation_w) as i32
                    - args.padding_w as i32;
                if y >= 0 && x >= 0 && y < height as i32 && x < width as i32 {
                    let image_ptr = image_idx + y as u32 * width + x as u32;
                    columns[col_pos] = image[image_ptr];
                } else {
                    columns[col_pos] = F::new(0.0)
                };
            } else {
                let y = out_y * args.stride_h + kernel_y * args.dilation_h;
                let x = out_x * args.stride_w + kernel_x * args.dilation_w;
                let image_ptr = image_idx + y * image.stride(2) + x * image.stride(3);
                columns[col_pos] = image[image_ptr];
            }
        }
    }
}

#[cfg(not(test))]
pub(crate) fn batches_per_run(batch_size: usize, out_h: usize, out_w: usize) -> Option<usize> {
    let cube_count_per_batch = (out_h * out_w).div_ceil(burn_common::PLANE_DIM_APPROX);
    let max_cube_count = u16::MAX as usize;
    let max_simultaneous = (max_cube_count / cube_count_per_batch).min(batch_size);
    if max_simultaneous == 0 {
        return None;
    }
    Some(
        (0..=max_simultaneous)
            .rev()
            .find(|per_run| batch_size % per_run == 0)
            .expect("Logically not possible"),
    )
}

#[cfg(test)]
#[allow(unused)]
pub(crate) fn batches_per_run(batch_size: usize, out_h: usize, out_w: usize) -> Option<usize> {
    Some(1)
}

fn im2col<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    options: ConvOptions<2>,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
) -> JitTensor<R> {
    let input = into_contiguous(input);
    let [batch_size, in_channels, _, _] = input.shape.dims();

    let col_shape_0 = in_channels * kernel_h * kernel_w;
    let col_shape_1 = batch_size * out_h * out_w;
    let shape_col = Shape::new([col_shape_0, col_shape_1]);
    let columns = empty_device::<R, E>(
        input.client.clone(),
        input.device.clone(),
        shape_col.clone(),
    );

    let num_elems = in_channels * batch_size * out_h * out_w;
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);

    let kernel_w_unroll = (kernel_w <= 8).then_some(kernel_w as u32);

    let vectorization = 1;

    unsafe {
        im2col_kernel::launch_unchecked::<E, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_handle_ref().as_tensor_arg(vectorization),
            columns.as_handle_ref().as_tensor_arg(vectorization),
            Im2ColArgsLaunch::new(
                ScalarArg::new(options.stride[0] as u32),
                ScalarArg::new(options.stride[1] as u32),
                ScalarArg::new(options.dilation[0] as u32),
                ScalarArg::new(options.dilation[1] as u32),
                ScalarArg::new(options.padding[0] as u32),
                ScalarArg::new(options.padding[1] as u32),
                ScalarArg::new(kernel_h as u32),
                ScalarArg::new(kernel_w as u32),
                ScalarArg::new(out_h as u32),
                ScalarArg::new(out_w as u32),
                ScalarArg::new(col_shape_1 as u32),
                ScalarArg::new(num_elems as u32),
            ),
            kernel_w_unroll,
            options.padding != [0, 0],
        )
    };

    columns
}

/// Perform a 2D convolution using the GEMM (im2col) algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
///
pub fn conv2d_im2col<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    options: ConvOptions<2>,
) -> Result<JitTensor<R>, ConvLaunchError> {
    let [batch_size, in_channels, in_height, in_width] = input.shape.dims();
    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims();
    let groups = options.groups;
    let out_c_per_group = out_channels / groups;

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

    if kernel_h == 1 && kernel_w == 1 && in_height == out_h && in_width == out_w {
        // Special case for 1x1 kernels (sometimes used to scale the image by a set of weights)
        return execute_1x1_kernel::<R, E>(input, weight, bias, options);
    }

    let batches_per_run = batches_per_run(batch_size, out_h, out_w)
        .expect("Image too large to run even one batch at once");
    let matmul_shape = Shape::new([groups, out_c_per_group, batches_per_run * out_h * out_w]);

    let mut out = if batches_per_run != batch_size {
        let runs = batch_size / batches_per_run;
        let out_shape = Shape::new([runs, out_channels, batches_per_run, out_h, out_w]);
        let out = empty_device::<R, E>(input.client.clone(), input.device.clone(), out_shape);
        let in_shape = Shape::new([runs, batches_per_run, in_channels, in_height, in_width]);
        let input = reshape(input, in_shape);
        let in_shape_run = Shape::new([batches_per_run, in_channels, in_height, in_width]);
        for run in 0..runs {
            let input = index::<R, E>(input.clone(), run);
            let input = reshape(input, in_shape_run.clone());
            let out_slice = index::<R, E>(out.clone(), run);
            let out_slice = reshape(out_slice, matmul_shape.clone());
            execute::<R, E>(
                input,
                weight.clone(),
                out_slice,
                options.clone(),
                out_h,
                out_w,
            )?;
        }
        let out = swap_dims(out, 1, 2);
        reshape(out, Shape::new([batch_size, out_channels, out_h, out_w]))
    } else {
        let out = empty_device::<R, E>(input.client.clone(), input.device.clone(), matmul_shape);
        execute::<R, E>(input, weight, out.clone(), options, out_h, out_w)?;
        let out = reshape(out, Shape::new([out_channels, batch_size, out_h, out_w]));
        swap_dims(out, 0, 1)
    };

    if let Some(bias) = bias {
        let bias = reshape(bias, Shape::new([1, out_channels, 1, 1]));
        out = launch_binop::<R, E, AddOp>(out, bias)
    }

    Ok(out)
}

fn execute_1x1_kernel<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    options: ConvOptions<2>,
) -> Result<JitTensor<R>, ConvLaunchError> {
    let [batch_size, _, height, width] = input.shape.dims();
    let [out_channels, in_c_per_grp, _, _] = weight.shape.dims();
    let groups = options.groups;
    let out_c_per_grp = out_channels / groups;

    let input = swap_dims(input, 0, 1); // [CNHW]

    let weight = reshape(weight, Shape::new([groups, out_c_per_grp, in_c_per_grp]));
    let in_shape = Shape::new([groups, in_c_per_grp, batch_size * height * width]);
    let input = reshape(input, in_shape);
    let out = matmul::<R, E>(weight, input, None, MatmulStrategy::default())?;
    let mut out = reshape(out, Shape::new([out_channels, batch_size, height, width]));

    if let Some(bias) = bias {
        let bias = reshape(bias, Shape::new([out_channels, 1, 1, 1]));
        out = launch_binop::<R, E, AddOp>(out, bias)
    }

    Ok(swap_dims(out, 0, 1))
}

fn execute<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    weight: JitTensor<R>,
    out: JitTensor<R>,
    options: ConvOptions<2>,
    out_h: usize,
    out_w: usize,
) -> Result<(), ConvLaunchError> {
    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims();
    let groups = options.groups;

    let columns = im2col::<R, E>(input, options.clone(), kernel_h, kernel_w, out_h, out_w);
    let [col_shape_0, col_shape_1] = columns.shape.dims();
    let col_shape_0 = col_shape_0 / groups;
    let out_c_per_group = out_channels / groups;

    let columns = reshape(columns, Shape::new([groups, col_shape_0, col_shape_1]));
    let weight = reshape(weight, Shape::new([groups, out_c_per_group, col_shape_0]));

    matmul::<R, E>(weight, columns, Some(out), Default::default())?;

    Ok(())
}
