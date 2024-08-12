use cubecl::{calculate_cube_count_elemwise, prelude::*};

use burn_tensor::{
    ops::{
        conv::calculate_conv_output_size, DeformConv2dBackward, DeformConvOptions,
        FloatTensorOps as _,
    },
    Shape, TensorData,
};

use crate::{
    kernel::into_contiguous,
    ops::numeric::{ones_device, zeros_device},
    tensor::JitTensor,
    FloatElement, IntElement, JitBackend, JitElement, JitRuntime,
};

#[derive(CubeLaunch)]
struct DeformConv2dArgs {
    conv_stride_h: UInt,
    conv_stride_w: UInt,
    dilation_h: UInt,
    dilation_w: UInt,
    padding_h: UInt,
    padding_w: UInt,
    offset_groups: UInt,
    kernel_height: UInt,
    kernel_width: UInt,

    batch_size: UInt,
    in_channels: UInt,
    height: UInt,
    width: UInt,
    out_h: UInt,
    out_w: UInt,
}

#[cube(launch)]
fn deform_image_to_column_kernel<F: Float>(
    input: &Tensor<F>,
    offset: &Tensor<F>,
    mask: &Tensor<F>,
    columns: &mut Tensor<F>,
    args: &DeformConv2dArgs,
    kernel_size_0_unroll: Comptime<Option<UInt>>,
    kernel_size_1_unroll: Comptime<Option<UInt>>,
) {
    // position shape: [in_channels, batch_size, out_h, out_w]
    // columns shape: [[in_channels, kernel_h, kernel_w], [batch_size, out_h, out_w]]

    if ABSOLUTE_POS >= columns.len() {
        return;
    }

    let kernel_height = Comptime::unwrap_or_else(kernel_size_0_unroll, || args.kernel_height);
    let unroll_y = Comptime::is_some(kernel_size_0_unroll);
    let kernel_width = Comptime::unwrap_or_else(kernel_size_1_unroll, || args.kernel_width);
    let unroll_x = Comptime::is_some(kernel_size_1_unroll);

    let out_h = args.out_h;
    let out_w = args.out_w;
    let batch_size = args.batch_size;
    let in_channels = args.in_channels;
    let height = args.height;
    let width = args.width;

    let out_x = ABSOLUTE_POS % out_w;
    let out_y = (ABSOLUTE_POS / out_w) % out_h;
    let out_batch = (ABSOLUTE_POS / (out_w * out_h)) % batch_size;
    let in_channel = ABSOLUTE_POS / (out_w * out_h * batch_size);
    let out_channel = in_channel * kernel_height * kernel_width;

    let channels_per_offset_group = args.in_channels / args.offset_groups;
    let group_index = in_channel / channels_per_offset_group;

    let mut col_base_idx = out_channel * (batch_size * out_h * out_w)
        + out_batch * (out_h * out_w)
        + out_y * out_w
        + out_x;

    let input_base_idx = out_batch * (in_channels * height * width) + in_channel * (height * width);

    let offset_base_idx = (out_batch * args.offset_groups + group_index)
        * 2
        * kernel_height
        * kernel_width
        * out_h
        * out_w;

    let mask_base_idx = (out_batch * args.offset_groups + group_index)
        * kernel_height
        * kernel_width
        * out_h
        * out_w;

    for kernel_y in range(0, kernel_height, unroll_y) {
        for kernel_x in range(0, kernel_width, unroll_x) {
            let mask_index = kernel_y * kernel_width + kernel_x;
            let offset_index = mask_index * 2;

            let mask_value =
                mask[mask_base_idx + mask_index * (out_h * out_w) + out_y * out_w + out_x];

            let offset_y =
                offset[offset_base_idx + offset_index * (out_h * out_w) + out_y * out_w + out_x];
            let offset_x = offset
                [offset_base_idx + (offset_index + 1) * (out_h * out_w) + out_y * out_w + out_x];
            let y = F::cast_from(
                (out_y * args.conv_stride_h - args.padding_h) + kernel_y * args.dilation_h,
            ) + offset_y;
            let x = F::cast_from(
                (out_x * args.conv_stride_w - args.padding_w) + kernel_x * args.dilation_w,
            ) + offset_x;

            let interpolated = bilinear_interpolate(
                input,
                F::cast_from(height),
                F::cast_from(width),
                y,
                x,
                input_base_idx,
            );

            columns[col_base_idx] = mask_value * interpolated;
            col_base_idx += batch_size * out_h * out_w;
        }
    }
}

#[cube]
fn bilinear_interpolate<F: Float>(
    input: &Tensor<F>,
    height: F,
    width: F,
    y: F,
    x: F,
    offset: UInt,
) -> F {
    let zero = F::new(0.0);
    let one = F::new(1.0);
    let neg_one = F::new(0.0 - 1.0);

    let mut result = zero;
    if y > neg_one && height > y && x > neg_one && width > x {
        let in_w = UInt::cast_from(width);

        let y_low = F::floor(y);
        let x_low = F::floor(x);
        let y_high = y_low + one;
        let x_high = x_low + one;

        let mut v1 = zero;
        let mut v2 = zero;
        let mut v3 = zero;
        let mut v4 = zero;
        if y_low >= 0.0 && x_low >= 0.0 {
            v1 = input[offset + UInt::cast_from(y_low) * in_w + UInt::cast_from(x_low)]
        };
        if y_low >= 0.0 && x_high < width {
            v2 = input[offset + UInt::cast_from(y_low) * in_w + UInt::cast_from(x_high)]
        };
        if y_high < height && x_low >= 0.0 {
            v3 = input[offset + UInt::cast_from(y_high) * in_w + UInt::cast_from(x_low)]
        };
        if y_high < height && x_high < width {
            v4 = input[offset + UInt::cast_from(y_high) * in_w + UInt::cast_from(x_high)]
        };

        let l_y = y - y_low;
        let l_x = x - x_low;
        let h_y = one - l_y;
        let h_x = one - l_x;

        let w1 = h_y * h_x;
        let w2 = h_y * l_x;
        let w3 = l_y * h_x;
        let w4 = l_y * l_x;

        result = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
    }
    result
}

fn deform_image_to_columns<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R, E, 4>,
    offset: JitTensor<R, E, 4>,
    mask: JitTensor<R, E, 4>,
    options: DeformConvOptions<2>,
    out_dims: (usize, usize),
    kernel_dims: (usize, usize),
) -> JitTensor<R, E, 2> {
    let [batch_size, in_channels, height, width] = input.shape.dims;
    let (out_height, out_width) = out_dims;
    let (kernel_height, kernel_width) = kernel_dims;

    let shape_out = Shape::new([
        in_channels * kernel_height * kernel_width,
        batch_size * out_height * out_width,
    ]);

    let output = zeros_device(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
    );

    let num_kernels = in_channels * batch_size * out_height * out_width;
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_kernels, cube_dim);

    deform_image_to_column_kernel::launch::<E::FloatPrimitive, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_handle_ref().as_tensor_arg(1),
        offset.as_handle_ref().as_tensor_arg(1),
        mask.as_handle_ref().as_tensor_arg(1),
        output.as_handle_ref().as_tensor_arg(1),
        DeformConv2dArgsLaunch::new(
            ScalarArg::new(options.stride[0] as u32),
            ScalarArg::new(options.stride[1] as u32),
            ScalarArg::new(options.dilation[0] as u32),
            ScalarArg::new(options.dilation[1] as u32),
            ScalarArg::new(options.padding[0] as u32),
            ScalarArg::new(options.padding[1] as u32),
            ScalarArg::new(options.offset_groups as u32),
            ScalarArg::new(kernel_height as u32),
            ScalarArg::new(kernel_width as u32),
            ScalarArg::new(batch_size as u32),
            ScalarArg::new(in_channels as u32),
            ScalarArg::new(height as u32),
            ScalarArg::new(width as u32),
            ScalarArg::new(out_height as u32),
            ScalarArg::new(out_width as u32),
        ),
        Some(kernel_height.into()),
        Some(kernel_width.into()),
    );

    output
}

pub(crate) fn deform_conv2d<R: JitRuntime, E: FloatElement, I: IntElement>(
    input: JitTensor<R, E, 4>,
    offset: JitTensor<R, E, 4>,
    weight: JitTensor<R, E, 4>,
    mask: Option<JitTensor<R, E, 4>>,
    bias: Option<JitTensor<R, E, 1>>,
    options: DeformConvOptions<2>,
) -> JitTensor<R, E, 4> {
    let device = JitBackend::<R, E, I>::float_device(&input);

    let input = into_contiguous(input);
    let offset = into_contiguous(offset);
    let weight = into_contiguous(weight);
    let mask = mask.map(|it| into_contiguous(it));
    let bias = bias.map(|it| into_contiguous(it));

    let [batch_size, in_channels, in_height, in_width] = input.shape.dims;
    let [out_channels, in_channels_per_group, weight_height, weight_width] = weight.shape.dims;

    debug_assert!(
        out_channels % options.weight_groups == 0,
        "Out channels must be divisible by weight groups"
    );
    debug_assert!(
        in_channels % options.weight_groups == 0,
        "In channels must be divisible by weight groups"
    );

    let out_height = calculate_conv_output_size(
        weight_height,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        in_height,
    );
    let out_width = calculate_conv_output_size(
        weight_width,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
        in_width,
    );

    let mask = mask.unwrap_or_else(|| {
        let shape = Shape::from([
            batch_size,
            options.offset_groups * weight_height * weight_width,
            out_height,
            out_width,
        ]);
        ones_device(input.client.clone(), input.device.clone(), shape)
    });

    let columns = deform_image_to_columns(
        input,
        offset,
        mask,
        options.clone(),
        (out_height, out_width),
        (weight_height, weight_width),
    );

    let [col_size_0, col_size_1] = JitBackend::<R, E, I>::float_shape(&columns).dims;

    let out_channels_per_weight_group = out_channels / options.weight_groups;

    let weight = JitBackend::<R, E, I>::float_reshape(
        weight,
        Shape::new([
            out_channels,
            in_channels_per_group * weight_height * weight_width,
        ]),
    );

    let mut output = JitBackend::<R, E, I>::float_zeros(
        Shape::new([
            options.weight_groups,
            out_channels_per_weight_group,
            batch_size * out_height * out_width,
        ]),
        &device,
    );

    let weight = JitBackend::<R, E, I>::float_reshape(
        weight,
        Shape::new([
            options.weight_groups,
            out_channels_per_weight_group,
            in_channels_per_group,
            weight_height,
            weight_width,
        ]),
    );
    let columns = JitBackend::<R, E, I>::float_reshape(
        columns,
        Shape::new([
            options.weight_groups,
            col_size_0 / options.weight_groups,
            col_size_1,
        ]),
    );

    for group in 0..options.weight_groups {
        let columns = JitBackend::<R, E, I>::float_reshape(
            JitBackend::<R, E, I>::float_narrow(columns.clone(), 0, group, 1),
            Shape::new([col_size_0 / options.weight_groups, col_size_1]),
        );
        let weight = JitBackend::<R, E, I>::float_reshape(
            JitBackend::<R, E, I>::float_narrow(weight.clone(), 0, group, 1),
            Shape::new([
                out_channels_per_weight_group,
                in_channels_per_group * weight_height * weight_width,
            ]),
        );
        let values = JitBackend::<R, E, I>::float_matmul(weight, columns);
        let values = JitBackend::<R, E, I>::float_reshape(
            values,
            Shape::new([
                1,
                out_channels_per_weight_group,
                batch_size * out_height * out_width,
            ]),
        );
        output = JitBackend::<R, E, I>::float_slice_assign(
            output,
            [
                group..group + 1,
                0..out_channels_per_weight_group,
                0..batch_size * out_height * out_width,
            ],
            values,
        );
    }

    let output = JitBackend::<R, E, I>::float_reshape(
        output,
        Shape::new([out_channels, batch_size, out_height, out_width]),
    );
    let output = JitBackend::<R, E, I>::float_swap_dims(output, 0, 1);

    if let Some(bias) = bias {
        let bias = JitBackend::<R, E, I>::float_reshape(bias, Shape::new([1, out_channels, 1, 1]));
        JitBackend::<R, E, I>::float_add(output, bias)
    } else {
        output
    }
}

fn debug_data<R: JitRuntime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
) -> TensorData {
    let bytes = tensor.client.read(tensor.handle.binding());
    TensorData::new(E::from_bytes(&bytes).to_vec(), tensor.shape)
}

/// Calculate the [deformable 2D convolution](crate::ops::ModuleOps::deform_conv2d) backward pass using convolutions.
#[allow(clippy::single_range_in_vec_init)]
pub(crate) fn deform_conv2d_backward<R: JitRuntime, E: FloatElement, I: IntElement>(
    input: JitTensor<R, E, 4>,
    offset: JitTensor<R, E, 4>,
    weight: JitTensor<R, E, 4>,
    mask: Option<JitTensor<R, E, 4>>,
    bias: Option<JitTensor<R, E, 1>>,
    grad_out: JitTensor<R, E, 4>,
    options: DeformConvOptions<2>,
) -> DeformConv2dBackward<JitBackend<R, E, I>> {
    type B<R, E, I> = JitBackend<R, E, I>;

    let weight_shape = B::<R, E, I>::float_shape(&weight);

    let [batch_size, _, out_h, out_w] = B::<R, E, I>::float_shape(&grad_out).dims;
    let [out_channels, in_channels_per_group, kernel_h, kernel_w] = weight_shape.dims;
    let n_weight_groups = options.weight_groups;
    let out_channels_per_group = out_channels / n_weight_groups;

    let gradient_bias = bias.map(|bias| {
        let grad = B::<R, E, I>::float_sum_dim(grad_out.clone(), 0);
        let grad = B::<R, E, I>::float_sum_dim(grad, 2);
        let grad = B::<R, E, I>::float_sum_dim(grad, 3);

        B::<R, E, I>::float_reshape(grad, B::<R, E, I>::float_shape(&bias))
    });

    let mut columns = zeros_device(
        grad_out.client.clone(),
        grad_out.device.clone(),
        Shape::new([
            n_weight_groups,
            in_channels_per_group * kernel_h * kernel_w,
            batch_size * out_h * out_w,
        ]),
    );
    let [col_size_0, col_size_1, col_size_2] = columns.shape.dims;
    let out_grad = B::<R, E, I>::float_reshape(
        grad_out,
        Shape::new([
            batch_size,
            n_weight_groups,
            out_channels_per_group,
            out_h,
            out_w,
        ]),
    );
    let out_grad = B::<R, E, I>::float_permute(out_grad, [1, 2, 0, 3, 4]);

    let weight = B::<R, E, I>::float_reshape(
        weight,
        Shape::new([
            n_weight_groups,
            out_channels_per_group,
            in_channels_per_group,
            kernel_h,
            kernel_w,
        ]),
    );

    for group in 0..n_weight_groups {
        let out_grad = B::<R, E, I>::float_reshape(
            B::<R, E, I>::float_narrow(out_grad.clone(), 0, group, 1),
            Shape::new([out_channels_per_group, batch_size * out_h * out_w]),
        );
        let weight = B::<R, E, I>::float_reshape(
            B::<R, E, I>::float_narrow(weight.clone(), 0, group, 1),
            Shape::new([
                out_channels_per_group,
                in_channels_per_group * kernel_h * kernel_w,
            ]),
        );
        let weight = B::<R, E, I>::float_swap_dims(weight, 0, 1);
        let values = B::<R, E, I>::float_matmul(weight, out_grad);
        let values = B::<R, E, I>::float_reshape(values, Shape::new([1, col_size_1, col_size_2]));
        columns = B::<R, E, I>::float_slice_assign(
            columns,
            [group..group + 1, 0..col_size_1, 0..col_size_2],
            values,
        );
    }

    let columns =
        B::<R, E, I>::float_reshape(columns, Shape::new([col_size_0 * col_size_1, col_size_2]));

    let (input_gradient, offset_gradient, mask_gradient) = backward_gradient_inputs::<R, E, I>(
        input.clone(),
        offset.clone(),
        mask.clone(),
        columns,
        &options,
        (kernel_h, kernel_w),
    );

    let input = into_contiguous(input);
    let offset = into_contiguous(offset);
    let mask = mask.map(|it| into_contiguous(it));
    let output_grad = into_contiguous(out_grad);

    let weight_grad = compute_weight_grad::<R, E, I>(
        input,
        offset,
        mask,
        output_grad,
        options,
        (kernel_h, kernel_w),
        out_channels,
    );

    DeformConv2dBackward::new(
        input_gradient,
        offset_gradient,
        weight_grad,
        mask_gradient,
        gradient_bias,
    )
}

fn compute_weight_grad<R: JitRuntime, E: FloatElement, I: IntElement>(
    input: JitTensor<R, E, 4>,
    offset: JitTensor<R, E, 4>,
    mask: Option<JitTensor<R, E, 4>>,
    out_grad: JitTensor<R, E, 5>,
    options: DeformConvOptions<2>,
    kernel_dims: (usize, usize),
    out_channels: usize,
) -> JitTensor<R, E, 4> {
    let device = JitBackend::<R, E, I>::float_device(&out_grad);

    let [batch_size, in_channels, _, _] = JitBackend::<R, E, I>::float_shape(&input).dims;
    let [_, _, _, out_h, out_w] = JitBackend::<R, E, I>::float_shape(&out_grad).dims;
    let (kernel_h, kernel_w) = kernel_dims;

    let weight_groups = options.weight_groups;
    let out_channels_per_weight_group = out_channels / weight_groups;
    let in_channels_per_group = in_channels / weight_groups;

    let mask = mask.unwrap_or_else(|| {
        let shape = Shape::new([
            batch_size,
            options.offset_groups * kernel_h * kernel_w,
            out_h,
            out_w,
        ]);
        ones_device(out_grad.client.clone(), out_grad.device.clone(), shape)
    });

    let columns = deform_image_to_columns(
        input,
        offset,
        mask,
        options,
        (out_h, out_w),
        (kernel_h, kernel_w),
    );
    let [col_size_0, col_size_1] = JitBackend::<R, E, I>::float_shape(&columns).dims;

    let mut grad_weight = JitBackend::<R, E, I>::float_zeros(
        Shape::new([
            weight_groups,
            out_channels_per_weight_group,
            in_channels_per_group,
            kernel_h,
            kernel_w,
        ]),
        &device,
    );

    let columns = JitBackend::<R, E, I>::float_reshape(
        columns,
        Shape::new([weight_groups, col_size_0 / weight_groups, col_size_1]),
    );

    for group in 0..weight_groups {
        let grad_out_buf = JitBackend::<R, E, I>::float_reshape(
            JitBackend::<R, E, I>::float_narrow(out_grad.clone(), 0, group, group + 1),
            Shape::new([out_channels_per_weight_group, batch_size * out_h * out_w]),
        );
        let columns = JitBackend::<R, E, I>::float_reshape(
            JitBackend::<R, E, I>::float_narrow(columns.clone(), 0, group, group + 1),
            Shape::new([col_size_0 / weight_groups, col_size_1]),
        );
        let columns = JitBackend::<R, E, I>::float_swap_dims(columns, 0, 1);

        let values = JitBackend::<R, E, I>::float_matmul(grad_out_buf, columns);
        let values = JitBackend::<R, E, I>::float_reshape(
            values,
            Shape::new([
                1,
                out_channels_per_weight_group,
                in_channels_per_group,
                kernel_h,
                kernel_w,
            ]),
        );
        grad_weight = JitBackend::<R, E, I>::float_slice_assign(
            grad_weight,
            [
                group..group + 1,
                0..out_channels_per_weight_group,
                0..in_channels_per_group,
                0..kernel_h,
                0..kernel_w,
            ],
            values,
        );
    }
    JitBackend::<R, E, I>::float_reshape(
        grad_weight,
        Shape::new([out_channels, in_channels_per_group, kernel_h, kernel_w]),
    )
}

type InputGradients<R, E> = (
    JitTensor<R, E, 4>,
    JitTensor<R, E, 4>,
    Option<JitTensor<R, E, 4>>,
);

fn backward_gradient_inputs<R: JitRuntime, E: FloatElement, I: IntElement>(
    x: JitTensor<R, E, 4>,
    offset: JitTensor<R, E, 4>,
    mask: Option<JitTensor<R, E, 4>>,
    columns: JitTensor<R, E, 2>,
    options: &DeformConvOptions<2>,
    kernel_dims: (usize, usize),
) -> InputGradients<R, E> {
    let input_shape = x.shape.clone();
    let (offset_gradient, mask_gradient) = compute_offset_and_mask_gradient::<R, E, I>(
        columns.clone(),
        x,
        offset.clone(),
        mask.clone(),
        options,
        kernel_dims,
    );

    let input_gradient =
        compute_input_grad::<R, E, I>(columns, offset, mask, options, kernel_dims, input_shape);

    (input_gradient, offset_gradient, mask_gradient)
}

fn compute_offset_and_mask_gradient<R: JitRuntime, E: FloatElement, I: IntElement>(
    columns: JitTensor<R, E, 2>,
    image: JitTensor<R, E, 4>,
    offset: JitTensor<R, E, 4>,
    mask: Option<JitTensor<R, E, 4>>,
    options: &DeformConvOptions<2>,
    kernel_dims: (usize, usize),
) -> (JitTensor<R, E, 4>, Option<JitTensor<R, E, 4>>) {
    let [batch_size, kernel_size_times_two, out_height, out_width] = offset.shape.dims;
    let [_, in_channels, height, width] = image.shape.dims;
    let use_mask = mask.is_some();
    let (kernel_height, kernel_width) = kernel_dims;

    let mask = mask.unwrap_or_else(|| {
        let shape = Shape::new([batch_size, kernel_size_times_two / 2, out_height, out_width]);
        ones_device(offset.client.clone(), offset.device.clone(), shape)
    });

    let grad_offset = zeros_device(
        offset.client.clone(),
        offset.device.clone(),
        offset.shape.clone(),
    );
    let debug: JitTensor<R, E, 4> = zeros_device(
        offset.client.clone(),
        offset.device.clone(),
        offset.shape.clone(),
    );
    let grad_mask = zeros_device(mask.client.clone(), mask.device.clone(), mask.shape.clone());
    let offset_channels = 2 * kernel_height * kernel_width * options.offset_groups;

    let num_elements_offset = offset.shape.num_elements();
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elements_offset, cube_dim);

    deform_col2img_coord_kernel::launch::<E::FloatPrimitive, R>(
        &image.client,
        cube_count,
        cube_dim,
        image.as_handle_ref().as_tensor_arg(1),
        offset.as_handle_ref().as_tensor_arg(1),
        mask.as_handle_ref().as_tensor_arg(1),
        columns.as_handle_ref().as_tensor_arg(1),
        grad_offset.as_handle_ref().as_tensor_arg(1),
        grad_mask.as_handle_ref().as_tensor_arg(1),
        debug.as_handle_ref().as_tensor_arg(1),
        DeformConv2dCol2ImgCoordArgsLaunch::new(
            ScalarArg::new(options.stride[0] as u32),
            ScalarArg::new(options.stride[1] as u32),
            ScalarArg::new(options.dilation[0] as u32),
            ScalarArg::new(options.dilation[1] as u32),
            ScalarArg::new(options.padding[0] as u32),
            ScalarArg::new(options.padding[1] as u32),
            ScalarArg::new(options.offset_groups as u32),
            ScalarArg::new(kernel_height as u32),
            ScalarArg::new(kernel_width as u32),
            ScalarArg::new(out_height as u32),
            ScalarArg::new(out_width as u32),
            ScalarArg::new(in_channels as u32),
            ScalarArg::new(height as u32),
            ScalarArg::new(width as u32),
            ScalarArg::new(batch_size as u32),
            ScalarArg::new(offset_channels as u32),
            ScalarArg::new(if use_mask { 1 } else { 0 }),
        ),
    );

    println!("Input: {}", debug_data(image.clone()));
    println!("Mask: {}", debug_data(mask.clone()));
    println!("Columns: {}", debug_data(columns.clone()));
    println!("Offset gradient: {}", debug_data(grad_offset.clone()));
    println!("Debug: {}", debug_data(debug.clone()));

    let mask_gradient = if use_mask { Some(grad_mask) } else { None };
    (grad_offset, mask_gradient)
}

#[derive(CubeLaunch)]
struct DeformConv2dCol2ImgCoordArgs {
    stride_h: UInt,
    stride_w: UInt,
    dilation_h: UInt,
    dilation_w: UInt,
    pad_h: UInt,
    pad_w: UInt,
    offset_groups: UInt,
    kernel_height: UInt,
    kernel_width: UInt,

    out_h: UInt,
    out_w: UInt,
    in_channels: UInt,
    height: UInt,
    width: UInt,
    batch_size: UInt,

    offset_channels: UInt,
    use_mask: UInt,
}

#[cube(launch)]
fn deform_col2img_coord_kernel<F: Float>(
    image: &Tensor<F>,
    offset: &Tensor<F>,
    mask: &Tensor<F>,
    columns: &Tensor<F>,
    grad_offset: &mut Tensor<F>,
    grad_mask: &mut Tensor<F>,
    debug: &mut Tensor<F>,
    args: &DeformConv2dCol2ImgCoordArgs,
) {
    // Position format: [batch, [kernel_h, kernel_w, 2], out_h, out_w]
    // Alternatively : [batch, offset_channels, out_h, out_w]

    if ABSOLUTE_POS > grad_offset.len() {
        return;
    }

    let out_w = args.out_w;
    let out_h = args.out_h;
    let weight_w = args.kernel_width;
    let weight_h = args.kernel_height;
    let n_offset_groups = args.offset_groups;
    let width = args.width;
    let batch_size = args.batch_size;
    let height = args.height;
    let offset_channels = args.offset_channels;

    let mut grad_offset_val = F::new(0.0);
    let mut grad_mask_val = F::new(0.0);

    let w = ABSOLUTE_POS % out_w;
    let h = (ABSOLUTE_POS / out_w) % out_h;
    let w_w = (ABSOLUTE_POS / (out_w * out_h * 2)) % weight_w;
    let w_h = (ABSOLUTE_POS / (out_w * out_h * 2 * weight_w)) % weight_h;
    let c = (ABSOLUTE_POS / (out_w * out_h)) % offset_channels;
    let b = ABSOLUTE_POS / (out_w * out_h * offset_channels);

    let offset_group = c / (weight_h * weight_w * 2);
    let col_step = weight_h * weight_w;

    let channels_per_offset_group = args.in_channels / args.offset_groups;

    let col_base_idx =
        offset_group * channels_per_offset_group * weight_h * weight_w * batch_size * out_w * out_h;
    let mut image_base_idx =
        (b * n_offset_groups + offset_group) * channels_per_offset_group * height * width;
    let offset_base_idx =
        (b * n_offset_groups + offset_group) * 2 * weight_h * weight_w * out_h * out_w;
    let mask_base_idx = (b * n_offset_groups + offset_group) * weight_h * weight_w * out_h * out_w;

    let offset_c = c - offset_group * 2 * weight_h * weight_w;
    let is_y_direction = offset_c % UInt::new(2) == 0;

    let c_bound = channels_per_offset_group * weight_h * weight_w;
    let mut col_c = offset_c / 2;

    // TODO: Fix once cubecl is fixed
    while col_c >= c_bound {
        let col_pos = (((col_c * batch_size + b) * out_h) + h) * out_w + w;

        // TODO: Investigate why it's recalculating stuff we already know
        let out_x = col_pos % out_w;
        let out_y = (col_pos / out_w) % out_h;
        let j = (col_pos / (out_w * out_h * batch_size)) % weight_w;
        let i = (col_pos / (out_w * out_h * batch_size * weight_w)) % weight_h;

        let mask_idx = i * weight_w + j;
        let offset_idx = mask_idx * 2;

        let offset_y_idx = (offset_idx * out_h + out_y) * out_w + out_x;
        let offset_x_idx = ((offset_idx + 1) * out_h + out_y) * out_w + out_x;

        let offset_y = offset[offset_base_idx + offset_y_idx];
        let offset_x = offset[offset_base_idx + offset_x_idx];

        let mask_value = mask[mask_base_idx + (mask_idx * out_h + out_y) * out_w + out_x];

        let y = F::cast_from((out_y * args.stride_h - args.pad_h) + i * args.dilation_h) + offset_y;
        let x = F::cast_from((out_x * args.stride_w - args.pad_w) + j * args.dilation_w) + offset_x;

        let weight = get_coordinate_weight(
            image,
            F::cast_from(height),
            F::cast_from(width),
            y,
            x,
            image_base_idx,
            is_y_direction,
        );

        debug[ABSOLUTE_POS] = F::cast_from(col_pos);

        grad_offset_val += mask_value * weight * columns[col_base_idx + col_pos];

        // Need to branch anyways, so check if mask is None
        if args.use_mask == 1 && is_y_direction {
            grad_mask_val += columns[col_base_idx + col_pos]
                * bilinear_interpolate(
                    image,
                    F::cast_from(height),
                    F::cast_from(width),
                    y,
                    x,
                    image_base_idx,
                );
        }

        image_base_idx += height * width;
        col_c += col_step;
    }

    grad_offset[ABSOLUTE_POS] = grad_offset_val;

    if args.use_mask == 1 {
        let idx = ((((b * n_offset_groups + offset_group) * weight_h + w_h) * weight_w + w_w)
            * out_h
            + h)
            * out_w
            + w;
        grad_mask[idx] = grad_mask_val
    }
}

#[cube]
fn get_coordinate_weight<F: Float>(
    input: &Tensor<F>,
    height: F,
    width: F,
    y: F,
    x: F,
    offset: UInt,
    is_y_direction: bool,
) -> F {
    let zero = F::new(0.0);
    let one = F::new(1.0);

    let stride_y = UInt::cast_from(width);

    let y_low = F::floor(y);
    let x_low = F::floor(x);
    let y_high = y_low + 1.0;
    let x_high = x_low + 1.0;

    let valid_y_low = y_low >= 0.0 && y_low < height;
    let valid_y_high = y_high >= 0.0 && y_high < height;
    let valid_x_low = x_low >= 0.0 && x_low < width;
    let valid_x_high = x_high >= 0.0 && x_high < width;

    let mut bottom_left = zero;
    let mut bottom_right = zero;
    let mut top_left = zero;
    let mut top_right = zero;
    if valid_y_low && valid_x_low {
        bottom_left = input[offset + UInt::cast_from(y_low) * stride_y + UInt::cast_from(x_low)];
    }
    if valid_y_low && valid_x_high {
        bottom_right = input[offset + UInt::cast_from(y_low) * stride_y + UInt::cast_from(x_high)];
    }
    if valid_y_high && valid_x_low {
        top_left = input[offset + UInt::cast_from(y_high) * stride_y + UInt::cast_from(x_low)];
    }
    if valid_y_high && valid_x_high {
        top_right = input[offset + UInt::cast_from(y_high) * stride_y + UInt::cast_from(x_high)];
    }

    #[allow(unused_assignments)]
    let mut result = zero;
    if is_y_direction {
        let delta_x = x - x_low;
        result = delta_x * (top_right - bottom_right) + (one - delta_x) * (top_left - bottom_left);
    } else {
        let delta_y = y - y_low;
        result = delta_y * (top_right - top_left) + (one - delta_y) * (bottom_right - bottom_left);
    }

    result
}

fn compute_input_grad<R: JitRuntime, E: FloatElement, I: IntElement>(
    columns: JitTensor<R, E, 2>,
    offset: JitTensor<R, E, 4>,
    mask: Option<JitTensor<R, E, 4>>,
    options: &DeformConvOptions<2>,
    kernel_dims: (usize, usize),
    input_shape: Shape<4>,
) -> JitTensor<R, E, 4> {
    let [_, in_channels, height, width] = input_shape.dims;
    let [batch_size, kernel_size_times_two, out_height, out_width] = offset.shape.dims;
    let (kernel_height, kernel_width) = kernel_dims;

    let grad_in = zeros_device::<R, E, 4>(
        offset.client.clone(),
        offset.device.clone(),
        Shape::new([batch_size, in_channels, height, width]),
    );

    let mask = mask.unwrap_or_else(|| {
        let shape = Shape::from([batch_size, kernel_size_times_two / 2, out_height, out_width]);
        ones_device(offset.client.clone(), offset.device.clone(), shape)
    });

    let num_elements = columns.shape.num_elements();
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elements, cube_dim);

    deform_col2img_kernel::launch::<E::FloatPrimitive, R>(
        &offset.client,
        cube_count,
        cube_dim,
        offset.as_handle_ref().as_tensor_arg(1),
        mask.as_handle_ref().as_tensor_arg(1),
        columns.as_handle_ref().as_tensor_arg(1),
        grad_in.as_handle_ref().as_tensor_arg(1),
        DeformConv2dCol2ImgArgsLaunch::new(
            ScalarArg::new(options.stride[0] as u32),
            ScalarArg::new(options.stride[1] as u32),
            ScalarArg::new(options.dilation[0] as u32),
            ScalarArg::new(options.dilation[1] as u32),
            ScalarArg::new(options.padding[0] as u32),
            ScalarArg::new(options.padding[1] as u32),
            ScalarArg::new(options.offset_groups as u32),
            ScalarArg::new(out_width as u32),
            ScalarArg::new(out_height as u32),
            ScalarArg::new(in_channels as u32),
            ScalarArg::new(height as u32),
            ScalarArg::new(width as u32),
            ScalarArg::new(kernel_height as u32),
            ScalarArg::new(kernel_width as u32),
            ScalarArg::new(batch_size as u32),
        ),
    );

    grad_in
}

#[derive(CubeLaunch)]
struct DeformConv2dCol2ImgArgs {
    stride_h: UInt,
    stride_w: UInt,
    dilation_h: UInt,
    dilation_w: UInt,
    pad_h: UInt,
    pad_w: UInt,
    offset_groups: UInt,
    out_w: UInt,
    out_h: UInt,
    in_channels: UInt,
    height: UInt,
    width: UInt,
    kernel_height: UInt,
    kernel_width: UInt,
    batch_size: UInt,
}

#[cube(launch)]
fn deform_col2img_kernel<F: Float>(
    offset: &Tensor<F>,
    mask: &Tensor<F>,
    columns: &Tensor<F>,
    grad_input: &mut Tensor<AtomicUInt>,
    args: &DeformConv2dCol2ImgArgs,
) {
    // Position format: [[in_channels, kernel_h, kernel_w], [batch_size, out_h, out_w]]

    let n_in_channels = args.in_channels;
    let height = args.height;
    let width = args.width;
    let out_h = args.out_h;
    let out_w = args.out_w;
    let kernel_h = args.kernel_height;
    let kernel_w = args.kernel_width;
    let n_offset_groups = args.offset_groups;
    let batch_size = args.batch_size;

    let out_x = ABSOLUTE_POS % out_w;
    let out_y = (ABSOLUTE_POS / out_w) % out_h;
    let batch = (ABSOLUTE_POS / (out_w * out_h)) % batch_size;
    let kernel_x = (ABSOLUTE_POS / (out_w * out_h * batch_size)) % kernel_w;
    let kernel_y = (ABSOLUTE_POS / (out_w * out_h * batch_size * kernel_w)) % kernel_h;
    let in_channel = ABSOLUTE_POS / (out_w * out_h * batch_size * kernel_w * kernel_h);

    let channels_per_offset_group = n_in_channels / n_offset_groups;
    let offset_group = in_channel / channels_per_offset_group;

    let offset_base_idx =
        (batch * n_offset_groups + offset_group) * 2 * kernel_h * kernel_w * out_h * out_w;
    let mask_base_idx =
        (batch * n_offset_groups + offset_group) * kernel_h * kernel_w * out_h * out_w;

    let mask_idx = kernel_y * kernel_w + kernel_x;
    let offset_idx = mask_idx * 2;

    let offset_y_idx = (offset_idx * out_h + out_y) * out_w + out_x;
    let offset_x_idx = ((offset_idx + 1) * out_h + out_y) * out_w + out_x;

    let offset_y = offset[offset_base_idx + offset_y_idx];
    let offset_x = offset[offset_base_idx + offset_x_idx];

    let mask_value = mask[mask_base_idx + (mask_idx * out_h + out_y) * out_w + out_x];

    let y =
        F::cast_from((out_y * args.stride_h - args.pad_h) + kernel_y * args.dilation_h) + offset_y;
    let x =
        F::cast_from((out_x * args.stride_w - args.pad_w) + kernel_x * args.dilation_w) + offset_x;

    for i in range(0, 3, Comptime::new(false)) {
        let dy = F::cast_from(i) - 1.0;
        for j in range(0, 3, Comptime::new(false)) {
            let dx = F::cast_from(j) - 1.0;

            let yp = F::floor(y) + dy;
            let xp = F::floor(x) + dx;
            let f_height = F::cast_from(height);
            let f_width = F::cast_from(width);

            if yp >= 0.0
                && yp < f_height
                && xp >= 0.0
                && xp < f_width
                && F::abs(y - yp) < 1.0
                && F::abs(x - xp) < 1.0
            {
                let y_point = UInt::cast_from(yp);
                let x_point = UInt::cast_from(xp);

                let gradient_pos =
                    ((batch * n_in_channels + in_channel) * height + y_point) * width + x_point;
                let weight = (F::new(1.0) - F::abs(y - yp)) * (F::new(1.0) - F::abs(x - xp));

                let value = mask_value * weight * columns[ABSOLUTE_POS];

                float_atomic_add::<F>(&grad_input[gradient_pos], value);
            }
        }
    }
}

#[cube]
fn float_atomic_add<F: Float>(ptr: &AtomicUInt, value: F) {
    if value == 0.0 {
        return;
    }

    // Bug in cubecl inverts while condition, temporary fix
    let mut done = false;
    while done {
        let v = AtomicUInt::load(ptr);
        let v_float = F::bitcast_from(v);
        let new = UInt::bitcast_from(v_float + value);
        done = AtomicUInt::compare_and_swap(ptr, v, new) == v;
    }
}
