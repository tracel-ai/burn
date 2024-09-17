use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions, FloatTensorOps as _},
    Shape,
};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{
    kernel::into_contiguous,
    ops::{numeric::empty_device, reshape, swap_dims},
    tensor::JitTensor,
    FloatElement, IntElement, JitBackend, JitRuntime,
};

use super::index;

#[derive(CubeLaunch)]
struct Im2ColArgs {
    stride_h: u32,
    stride_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    padding_h: i32,
    padding_w: i32,

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

            let y = (out_y * args.stride_h + kernel_y * args.dilation_h) as i32 - args.padding_h;
            let x = (out_x * args.stride_w + kernel_x * args.dilation_w) as i32 - args.padding_w;
            if y >= 0 && x >= 0 && y < height as i32 && x < width as i32 {
                let image_ptr = image_idx + y as u32 * width + x as u32;
                columns[col_pos] = image[image_ptr];
            } else {
                columns[col_pos] = F::new(0.0)
            };
        }
    }
}

fn im2col<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R, E, 4>,
    options: ConvOptions<2>,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
) -> JitTensor<R, E, 2> {
    let input = into_contiguous(input);
    let [batch_size, in_channels, _, _] = input.shape.dims;

    let col_shape_0 = in_channels * kernel_h * kernel_w;
    let col_shape_1 = batch_size * out_h * out_w;
    let shape_col = Shape::new([col_shape_0, col_shape_1]);
    let columns = empty_device(
        input.client.clone(),
        input.device.clone(),
        shape_col.clone(),
    );

    let num_elems = in_channels * batch_size * out_h * out_w;
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems, cube_dim);

    let kernel_w_unroll = (kernel_w <= 8).then_some(kernel_w as u32);

    unsafe {
        im2col_kernel::launch_unchecked::<E, R>(
            &input.client,
            cube_count,
            cube_dim,
            input.as_handle_ref().as_tensor_arg(1),
            columns.as_handle_ref().as_tensor_arg(1),
            Im2ColArgsLaunch::new(
                ScalarArg::new(options.stride[0] as u32),
                ScalarArg::new(options.stride[1] as u32),
                ScalarArg::new(options.dilation[0] as u32),
                ScalarArg::new(options.dilation[1] as u32),
                ScalarArg::new(options.padding[0] as i32),
                ScalarArg::new(options.padding[1] as i32),
                ScalarArg::new(kernel_h as u32),
                ScalarArg::new(kernel_w as u32),
                ScalarArg::new(out_h as u32),
                ScalarArg::new(out_w as u32),
                ScalarArg::new(col_shape_1 as u32),
                ScalarArg::new(num_elems as u32),
            ),
            kernel_w_unroll,
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
pub fn conv2d_im2col<R: JitRuntime, E: FloatElement, I: IntElement>(
    input: JitTensor<R, E, 4>,
    weight: JitTensor<R, E, 4>,
    bias: Option<JitTensor<R, E, 1>>,
    options: ConvOptions<2>,
) -> JitTensor<R, E, 4> {
    let [batch_size, _, in_height, in_width] = input.shape.dims;
    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims;
    let groups = options.groups;

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

    let columns = im2col(input, options.clone(), kernel_h, kernel_w, out_h, out_w);
    let [col_shape_0, col_shape_1] = columns.shape.dims;

    let out = if groups > 1 {
        let col_shape_0 = col_shape_0 / groups;
        let out_c_per_group = out_channels / groups;

        let columns = reshape(columns, Shape::new([groups, col_shape_0, col_shape_1]));
        let weight = reshape(weight, Shape::new([groups, out_c_per_group, col_shape_0]));

        let shape_out = Shape::new([groups, out_c_per_group, col_shape_1]);
        let mut out = empty_device(columns.client.clone(), columns.device.clone(), shape_out);

        for group in 0..groups {
            let weight = index::<R, E, I>(weight.clone(), group);
            let columns = index::<R, E, I>(columns.clone(), group);
            let values = JitBackend::<R, E, I>::float_matmul(weight, columns);
            let values = reshape(values, Shape::new([1, out_c_per_group, col_shape_1]));
            out = JitBackend::<R, E, I>::float_slice_assign(
                out,
                [group..group + 1, 0..out_c_per_group, 0..col_shape_1],
                values,
            )
        }
        reshape(out, Shape::new([out_channels, batch_size, out_h, out_w]))
    } else {
        let weight = reshape(weight, Shape::new([out_channels, col_shape_0]));
        let out = JitBackend::<R, E, I>::float_matmul(weight, columns);
        reshape(out, Shape::new([out_channels, batch_size, out_h, out_w]))
    };

    let mut out = swap_dims(out, 0, 1);
    if let Some(bias) = bias {
        let bias = reshape(bias, Shape::new([1, out_channels, 1, 1]));
        out = JitBackend::<R, E, I>::float_add(out, bias)
    }
    out
}
