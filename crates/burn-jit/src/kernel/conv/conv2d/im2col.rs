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
    stride_h: UInt,
    stride_w: UInt,
    dilation_h: UInt,
    dilation_w: UInt,
    padding_h: I32,
    padding_w: I32,

    kernel_h: UInt,
    kernel_w: UInt,
    out_h: UInt,
    out_w: UInt,

    batch_size: UInt,
    in_channels: UInt,
    height: UInt,
    width: UInt,
    col_size_1: UInt,

    num_elements: UInt,
}

#[cube(launch_unchecked)]
fn im2col_kernel<F: Float>(
    image: &Tensor<F>,
    columns: &mut Tensor<F>,
    args: &Im2ColArgs,
    kernel_h_unroll: Comptime<Option<UInt>>,
    kernel_w_unroll: Comptime<Option<UInt>>,
) {
    // position shape: [in_channels, batch_size, out_h, out_w]
    // columns shape: [[in_channels, kernel_h, kernel_w], [batch_size, out_h, out_w]]

    let batch_size = args.batch_size;
    let in_channels = args.in_channels;
    let height = args.height;
    let width = args.width;

    let out_h = args.out_h;
    let out_w = args.out_w;

    if ABSOLUTE_POS > args.num_elements {
        return;
    }

    let out_x = ABSOLUTE_POS % out_w;
    let out_y = ABSOLUTE_POS / out_w % out_h;
    let batch = ABSOLUTE_POS / (out_w * out_h) % batch_size;
    let channel = ABSOLUTE_POS / (out_w * out_h * batch_size) % in_channels;

    let kernel_h = Comptime::unwrap_or_else(kernel_h_unroll, || args.kernel_h);
    let unroll_h = Comptime::is_some(kernel_h_unroll);
    let kernel_w = Comptime::unwrap_or_else(kernel_w_unroll, || args.kernel_w);
    let unroll_w = Comptime::is_some(kernel_w_unroll);

    let image_idx = batch * in_channels * height * width + channel * height * width;
    let col_idx = channel * kernel_h * kernel_w * args.col_size_1
        + batch * out_h * out_w
        + out_y * out_w
        + out_x;
    let i_height = I32::cast_from(height);
    let i_width = I32::cast_from(width);

    for kernel_y in range(0, kernel_h, unroll_h) {
        for kernel_x in range(0, kernel_w, unroll_w) {
            let kernel_pos = kernel_y * kernel_w + kernel_x;
            let col_pos = col_idx + kernel_pos * args.col_size_1;

            let y =
                I32::cast_from(out_y * args.stride_h + kernel_y * args.dilation_h) - args.padding_h;
            let x =
                I32::cast_from(out_x * args.stride_w + kernel_x * args.dilation_w) - args.padding_w;
            if y >= 0 && x >= 0 && y < i_height && x < i_width {
                let image_y = UInt::cast_from(y);
                let image_x = UInt::cast_from(x);
                let image_ptr = image_idx + image_y * width + image_x;
                let pixel_value = image[image_ptr];
                columns[col_pos] = pixel_value;
            } else {
                columns[col_pos] = F::new(0.0);
            }
        }
    }
}

fn im2col<R: JitRuntime, E: FloatElement, I: IntElement>(
    input: JitTensor<R, E, 4>,
    options: ConvOptions<2>,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
) -> JitTensor<R, E, 2> {
    let input = into_contiguous(input);
    let [batch_size, in_channels, height, width] = input.shape.dims;

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

    unsafe {
        im2col_kernel::launch_unchecked::<E::FloatPrimitive, R>(
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
                ScalarArg::new(batch_size as u32),
                ScalarArg::new(in_channels as u32),
                ScalarArg::new(height as u32),
                ScalarArg::new(width as u32),
                ScalarArg::new(col_shape_1 as u32),
                ScalarArg::new(num_elems as u32),
            ),
            None,
            None,
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

    let columns = im2col::<R, E, I>(input, options.clone(), kernel_h, kernel_w, out_h, out_w);
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
