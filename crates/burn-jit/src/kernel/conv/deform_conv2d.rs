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
    ops::numeric::{empty_device, ones_device, zeros_device},
    tensor::JitTensor,
    FloatElement, IntElement, JitBackend, JitRuntime,
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
    columns2: &mut Tensor<F>,
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
            columns2[col_base_idx] = y;
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
        if y_low >= zero && x_low >= zero {
            v1 = input[offset + UInt::cast_from(y_low) * in_w + UInt::cast_from(x_low)]
        };
        if y_low >= zero && x_high <= width - one {
            v2 = input[offset + UInt::cast_from(y_low) * in_w + UInt::cast_from(x_high)]
        };
        if y_high <= height - one && x_low >= zero {
            v3 = input[offset + UInt::cast_from(y_high) * in_w + UInt::cast_from(x_low)]
        };
        if y_high <= height - one && x_high <= width - one {
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
    let out_2: JitTensor<R, E, 2> = zeros_device(
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
        TensorArg::new(&input.handle, &input.strides, &input.shape.dims),
        TensorArg::new(&offset.handle, &offset.strides, &offset.shape.dims),
        TensorArg::new(&mask.handle, &mask.strides, &mask.shape.dims),
        TensorArg::new(&output.handle, &output.strides, &output.shape.dims),
        TensorArg::new(&out_2.handle, &out_2.strides, &out_2.shape.dims),
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

    let mask = if let Some(mask) = mask {
        mask
    } else {
        let shape = Shape::from([
            batch_size,
            options.offset_groups * weight_height * weight_width,
            out_height,
            out_width,
        ]);
        ones_device(input.client.clone(), input.device.clone(), shape)
    };

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

fn debug_data<R: JitRuntime, E: FloatElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
) -> TensorData {
    let bytes = tensor.client.read(tensor.handle.binding());
    TensorData::new(E::from_bytes(&bytes).to_vec(), tensor.shape)
}

/// Calculate the [deformable 2D convolution](crate::ops::ModuleOps::deform_conv2d) backward pass using convolutions.
#[allow(clippy::single_range_in_vec_init)]
pub(crate) fn deform_conv2d_backward<R: JitRuntime, E: FloatElement, I: IntElement>(
    x: JitTensor<R, E, 4>,
    offset: JitTensor<R, E, 4>,
    weight: JitTensor<R, E, 4>,
    mask: Option<JitTensor<R, E, 4>>,
    bias: Option<JitTensor<R, E, 1>>,
    output_grad: JitTensor<R, E, 4>,
    options: DeformConvOptions<2>,
) -> DeformConv2dBackward<JitBackend<R, E, I>> {
    type B<R, E, I> = JitBackend<R, E, I>;

    let weight_shape = B::<R, E, I>::float_shape(&weight);
    let weight_device = B::<R, E, I>::float_device(&weight);

    let [batch_size, _channels_in, _, _] = B::<R, E, I>::float_shape(&x).dims;
    let [_, _, height_out, width_out] = B::<R, E, I>::float_shape(&output_grad).dims;
    let [channels_out, channels_in, kernel_size_1, kernel_size_2] = weight_shape.dims;

    let columns = {
        let weight = B::<R, E, I>::float_swap_dims::<2>(
            B::<R, E, I>::float_reshape(
                weight.clone(),
                Shape::new([channels_out, channels_in * kernel_size_1 * kernel_size_2]),
            ),
            0,
            1,
        );
        let out_grad = B::<R, E, I>::float_reshape(
            output_grad.clone(),
            Shape::new([batch_size, channels_out, height_out * width_out]),
        );
        let mut columns = B::<R, E, I>::float_empty(
            Shape::new([
                batch_size,
                channels_in * kernel_size_1 * kernel_size_2,
                height_out * width_out,
            ]),
            &weight_device,
        );
        for b in 0..batch_size {
            let out_grad = B::<R, E, I>::float_reshape(
                B::<R, E, I>::float_slice(out_grad.clone(), [b..b + 1]),
                Shape::new([channels_out, height_out * width_out]),
            );
            let values = B::<R, E, I>::float_reshape(
                B::<R, E, I>::float_matmul(weight.clone(), out_grad),
                Shape::new([1, channels_out, height_out * width_out]),
            );
            columns = B::<R, E, I>::float_slice_assign(columns, [b..b + 1], values);
        }
        B::<R, E, I>::float_reshape(
            columns,
            Shape::new([
                batch_size,
                channels_in,
                kernel_size_1 * kernel_size_2,
                height_out * width_out,
            ]),
        )
    };

    let gradient_bias = bias.map(|bias| {
        let grad = B::<R, E, I>::float_swap_dims(output_grad.clone(), 0, 1);
        let grad = B::<R, E, I>::float_reshape(
            grad,
            Shape::new([channels_out, batch_size * height_out * width_out]),
        );
        let grad = B::<R, E, I>::float_sum_dim(grad, 1);

        B::<R, E, I>::float_reshape(grad, B::<R, E, I>::float_shape(&bias))
    });

    let input = into_contiguous(x);
    let offset = into_contiguous(offset);
    let weight = into_contiguous(weight);
    let mask = mask.map(|it| into_contiguous(it));
    let columns = into_contiguous(columns);
    let output_grad = into_contiguous(output_grad);

    let (input_gradient, offset_gradient, mask_gradient) = backward_gradient_inputs(
        &input,
        &offset,
        &weight,
        mask.clone(),
        &output_grad,
        columns,
        &options,
    );

    let weight_grad =
        compute_weight_grad::<R, E, I>(input, offset, weight, mask, output_grad, options);

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
    weight: JitTensor<R, E, 4>,
    mask: Option<JitTensor<R, E, 4>>,
    out_grad: JitTensor<R, E, 4>,
    options: DeformConvOptions<2>,
) -> JitTensor<R, E, 4> {
    let device = JitBackend::<R, E, I>::float_device(&out_grad);

    let [batch_size, _, _, _] = JitBackend::<R, E, I>::float_shape(&input).dims;
    let [out_channels, in_channels_per_group, kernel_h, kernel_w] =
        JitBackend::<R, E, I>::float_shape(&weight).dims;
    let [_, _, out_h, out_w] = JitBackend::<R, E, I>::float_shape(&out_grad).dims;

    let weight_groups = options.weight_groups;
    let out_channels_per_weight_group = out_channels / weight_groups;

    let mask = if let Some(mask) = mask {
        mask
    } else {
        ones_device(
            out_grad.client.clone(),
            out_grad.device.clone(),
            Shape::new([
                batch_size,
                options.offset_groups * kernel_h * kernel_w,
                out_h,
                out_w,
            ]),
        )
    };

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
    let grad_out_buf = JitBackend::<R, E, I>::float_reshape(
        out_grad.clone(),
        Shape::new([
            batch_size,
            weight_groups,
            out_channels_per_weight_group,
            out_h,
            out_w,
        ]),
    );
    let grad_out_buf = into_contiguous(JitBackend::<R, E, I>::float_permute(
        grad_out_buf,
        [1, 2, 0, 3, 4],
    ));
    let columns = JitBackend::<R, E, I>::float_reshape(
        columns,
        Shape::new([weight_groups, col_size_0 / weight_groups, col_size_1]),
    );

    for group in 0..weight_groups {
        let grad_out_buf = JitBackend::<R, E, I>::float_reshape(
            JitBackend::<R, E, I>::float_narrow(grad_out_buf.clone(), 0, group, group + 1),
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

fn backward_gradient_inputs<R: JitRuntime, E: FloatElement>(
    x: &JitTensor<R, E, 4>,
    offset: &JitTensor<R, E, 4>,
    weight: &JitTensor<R, E, 4>,
    mask: Option<JitTensor<R, E, 4>>,
    output_grad: &JitTensor<R, E, 4>,
    columns: JitTensor<R, E, 4>,
    options: &DeformConvOptions<2>,
) -> InputGradients<R, E> {
    let (offset_gradient, mask_gradient) =
        compute_offset_and_mask_gradient(output_grad, x, offset, weight, mask.clone(), options);

    let input_gradient = compute_input_grad(x, &columns, offset, weight, mask, options);

    (input_gradient, offset_gradient, mask_gradient)
}

fn compute_offset_and_mask_gradient<R: JitRuntime, E: FloatElement>(
    output_gradient: &JitTensor<R, E, 4>,
    input: &JitTensor<R, E, 4>,
    offset: &JitTensor<R, E, 4>,
    weight: &JitTensor<R, E, 4>,
    mask: Option<JitTensor<R, E, 4>>,
    options: &DeformConvOptions<2>,
) -> (JitTensor<R, E, 4>, Option<JitTensor<R, E, 4>>) {
    let [batch_size, kernel_size_times_two, out_height, out_width] = offset.shape.dims;
    let [_, _, kernel_width, kernel_height] = weight.shape.dims;
    let use_mask = mask.is_some();

    let mask = if let Some(mask) = mask {
        mask
    } else {
        let shape = Shape::new([batch_size, kernel_size_times_two / 2, out_height, out_width]);
        ones_device(offset.client.clone(), offset.device.clone(), shape)
    };

    let offset_gradient = zeros_device(
        offset.client.clone(),
        offset.device.clone(),
        offset.shape.clone(),
    );
    let mask_gradient = zeros_device(mask.client.clone(), mask.device.clone(), mask.shape.clone());

    let num_elements_offset = offset.shape.num_elements();
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elements_offset, cube_dim);

    deform_col2img_coord_kernel::launch::<E::FloatPrimitive, R>(
        &input.client,
        cube_count,
        cube_dim,
        TensorArg::new(
            &output_gradient.handle,
            &output_gradient.strides,
            &output_gradient.shape.dims,
        ),
        TensorArg::new(&input.handle, &input.strides, &input.shape.dims),
        TensorArg::new(&offset.handle, &offset.strides, &offset.shape.dims),
        TensorArg::new(&mask.handle, &mask.strides, &mask.shape.dims),
        TensorArg::new(
            &output_gradient.handle,
            &output_gradient.strides,
            &output_gradient.shape.dims,
        ),
        TensorArg::new(
            &output_gradient.handle,
            &output_gradient.strides,
            &output_gradient.shape.dims,
        ),
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
        ),
    );

    let mask_gradient = if use_mask { Some(mask_gradient) } else { None };
    (offset_gradient, mask_gradient)
}

#[derive(CubeLaunch)]
struct DeformConv2dCol2ImgCoordArgs {
    conv_stride_h: UInt,
    conv_stride_w: UInt,
    dilation_h: UInt,
    dilation_w: UInt,
    padding_h: UInt,
    padding_w: UInt,
    offset_groups: UInt,
    kernel_height: UInt,
    kernel_width: UInt,
}

#[cube(launch)]
fn deform_col2img_coord_kernel<F: Float>(
    output_gradient: &Tensor<F>,
    input: &Tensor<F>,
    offset: &Tensor<F>,
    mask: &Tensor<F>,
    offset_gradient: &mut Tensor<F>,
    mask_gradient: &mut Tensor<F>,
    args: &DeformConv2dCol2ImgCoordArgs,
) {
    // Position format: [batch, [offset_group, kernel_y, kernel_x], out_y, out_x]
    if ABSOLUTE_POS > offset_gradient.len() {
        return;
    }

    let kernel_height = args.kernel_height;
    let kernel_width = args.kernel_width;

    let mut offset_gradient_sum = F::new(0.0);
    let mut mask_gradient_sum = F::new(0.0);

    let in_channels = input.shape(1);
    let height = input.shape(2);
    let width = input.shape(3);

    let batch = ABSOLUTE_POS / offset.stride(0) % offset.shape(0);
    let offset_channel = ABSOLUTE_POS / offset.stride(1) % offset.shape(1);
    let out_y = ABSOLUTE_POS / offset.stride(2) % offset.shape(2);
    let out_x = ABSOLUTE_POS / offset.stride(3) % offset.shape(3);

    let out_stride_batch = output_gradient.stride(0);
    let out_stride_channel = output_gradient.stride(1);
    let out_stride_y = output_gradient.stride(2);
    let out_stride_x = output_gradient.stride(3);

    let kernel_x = offset_channel / 2 % kernel_width;
    let kernel_y = offset_channel / (UInt::new(2) * kernel_width) % kernel_height;
    let kernel_stride_group = kernel_width * UInt::cast_from(kernel_height);
    let kernel_stride_y = kernel_width;
    let kernel_stride_x = UInt::new(1);

    let offset_group = offset_channel / (kernel_width * (kernel_height * UInt::new(2)));
    let column_step = UInt::cast_from(kernel_width) * kernel_height;

    let channels_per_offset_group = in_channels / args.offset_groups;
    let offset_channel = ABSOLUTE_POS / offset.stride(1) % offset.shape(1);
    let channel_start = offset_channel - offset_group * UInt::new(2) * kernel_height * kernel_width;
    let is_y_direction = channel_start % UInt::new(2) == 0;

    let offset_stride_batch = offset.stride(0);
    let offset_stride_index = offset.stride(1);
    let offset_stride_y = offset.stride(2);
    let offset_stride_x = offset.stride(3);

    let mask_stride_batch = mask.stride(0);
    let mask_stride_index = mask.stride(1);
    let mask_stride_y = mask.stride(2);
    let mask_stride_x = mask.stride(3);

    for out_channel in range(
        channel_start / UInt::new(2),
        channels_per_offset_group,
        Comptime::new(false),
    ) {
        let channel_offset = out_channel * column_step;
        let out_position = batch * out_stride_batch
            + channel_offset * out_stride_channel
            + out_y * out_stride_y
            + out_x * out_stride_x;

        let mask_index = offset_group * kernel_stride_group
            + kernel_y * kernel_stride_y
            + kernel_x * kernel_stride_x;
        let offset_index = mask_index * UInt::new(2);

        let offset_position =
            batch * offset_stride_batch + out_y * offset_stride_y + out_x * offset_stride_x;
        let offset_y = offset[offset_position + offset_index * offset_stride_index];
        let offset_x = offset[offset_position + (offset_index + 1) * offset_stride_index];

        let mask_value = mask[batch * mask_stride_batch
            + mask_index * mask_stride_index
            + out_y * mask_stride_y
            + out_x * mask_stride_x];

        let y = (out_y * args.conv_stride_h - args.padding_h) + kernel_y * args.dilation_h;
        let x = (out_x * args.conv_stride_w - args.padding_w) + kernel_x * args.dilation_w;

        let y = F::cast_from(y) + offset_y;
        let x = F::cast_from(x) + offset_x;

        let weight_interpolated = get_coordinate_weight(
            input,
            F::cast_from(height),
            F::cast_from(width),
            y,
            x,
            batch * input.stride(0) + offset_group * channels_per_offset_group * input.stride(1),
            is_y_direction,
        );
        offset_gradient_sum += mask_value * weight_interpolated * output_gradient[out_position];

        if is_y_direction {
            mask_gradient_sum += output_gradient[out_position]
                * bilinear_interpolate(
                    input,
                    F::cast_from(height),
                    F::cast_from(width),
                    y,
                    x,
                    batch * input.stride(0) + out_channel * input.stride(1),
                );
        }
    }

    offset_gradient[ABSOLUTE_POS] = offset_gradient_sum;

    if is_y_direction {
        let idx = batch * mask_stride_batch
            + (offset_group * kernel_stride_group
                + kernel_y * kernel_stride_y
                + kernel_x * kernel_stride_x)
                * mask_stride_index
            + out_y * mask_stride_y
            + out_x * mask_stride_x;
        mask_gradient[idx] = mask_gradient_sum;
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

    let stride_y = input.shape(2);
    let stride_x = input.shape(3);

    let y_low = F::floor(y);
    let x_low = F::floor(x);
    let y_high = y_low + one;
    let x_high = x_low + one;

    let valid_y_low = y_low >= zero && y_low < height;
    let valid_y_high = y_high >= zero && y_high < height;
    let valid_x_low = x_low >= zero && x_low < width;
    let valid_x_high = x_high >= zero && x_high < width;

    let mut bottom_left = zero;
    let mut bottom_right = zero;
    let mut top_left = zero;
    let mut top_right = zero;
    if valid_y_low && valid_x_low {
        bottom_left =
            input[offset + UInt::cast_from(y_low) * stride_y + UInt::cast_from(x_low) * stride_x];
    }
    if valid_y_low && valid_x_high {
        bottom_right =
            input[offset + UInt::cast_from(y_low) * stride_y + UInt::cast_from(x_high) * stride_x];
    }
    if valid_y_high && valid_x_low {
        top_left =
            input[offset + UInt::cast_from(y_high) * stride_y + UInt::cast_from(x_low) * stride_x];
    }
    if valid_y_high && valid_x_high {
        top_right =
            input[offset + UInt::cast_from(y_high) * stride_y + UInt::cast_from(x_high) * stride_x];
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

fn compute_input_grad<R: JitRuntime, E: FloatElement>(
    input: &JitTensor<R, E, 4>,
    columns: &JitTensor<R, E, 4>,
    offset: &JitTensor<R, E, 4>,
    weight: &JitTensor<R, E, 4>,
    mask: Option<JitTensor<R, E, 4>>,
    options: &DeformConvOptions<2>,
) -> JitTensor<R, E, 4> {
    let [_, _, height, width] = input.shape.dims;
    let [batch_size, kernel_size_times_two, out_height, out_width] = offset.shape.dims;
    let [_, _, kernel_height, kernel_width] = weight.shape.dims;

    let input_gradient = empty_device(
        input.client.clone(),
        input.device.clone(),
        input.shape.clone(),
    );

    let mask = if let Some(mask) = mask {
        mask
    } else {
        let shape = Shape::from([batch_size, kernel_size_times_two / 2, out_height, out_width]);
        ones_device(input.client.clone(), input.device.clone(), shape)
    };

    let num_elements = columns.shape.num_elements();
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elements, cube_dim);

    deform_col2img_kernel::launch::<E::FloatPrimitive, R>(
        &input.client,
        cube_count,
        cube_dim,
        TensorArg::new(&columns.handle, &columns.strides, &columns.shape.dims),
        TensorArg::new(&offset.handle, &offset.strides, &offset.shape.dims),
        TensorArg::new(&mask.handle, &mask.strides, &mask.shape.dims),
        TensorArg::new(
            &input_gradient.handle,
            &input_gradient.strides,
            &input_gradient.shape.dims,
        ),
        DeformConv2dCol2ImgArgsLaunch::new(
            ScalarArg::new(options.stride[0] as u32),
            ScalarArg::new(options.stride[1] as u32),
            ScalarArg::new(options.dilation[0] as u32),
            ScalarArg::new(options.dilation[1] as u32),
            ScalarArg::new(options.padding[0] as u32),
            ScalarArg::new(options.padding[1] as u32),
            ScalarArg::new(options.weight_groups as u32),
            ScalarArg::new(options.offset_groups as u32),
            ScalarArg::new(out_width as u32),
            ScalarArg::new(out_height as u32),
            ScalarArg::new(height as u32),
            ScalarArg::new(width as u32),
            ScalarArg::new(kernel_height as u32),
            ScalarArg::new(kernel_width as u32),
        ),
    );

    input_gradient
}

#[derive(CubeLaunch)]
struct DeformConv2dCol2ImgArgs {
    conv_stride_h: UInt,
    conv_stride_w: UInt,
    dilation_h: UInt,
    dilation_w: UInt,
    padding_h: UInt,
    padding_w: UInt,
    weight_groups: UInt,
    offset_groups: UInt,
    out_w: UInt,
    out_h: UInt,
    height: UInt,
    width: UInt,
    kernel_height: UInt,
    kernel_width: UInt,
}

#[cube(launch)]
fn deform_col2img_kernel<F: Float>(
    columns: &Tensor<F>,
    offset: &Tensor<F>,
    mask: &Tensor<F>,
    input_gradient: &mut Tensor<F>,
    args: &DeformConv2dCol2ImgArgs,
) {
    // Position format: [batch, in_channel, [kernel_y, kernel_x], [out_y, out_x]]

    if ABSOLUTE_POS > columns.len() {
        return;
    }

    let out_w = args.out_w;
    let out_h = args.out_h;

    let kernel_height = args.kernel_height;
    let kernel_width = args.kernel_width;

    let batch_size = columns.shape(0);
    let channels = columns.shape(1);

    let col_stride_batch = columns.stride(0);
    let col_stride_channel = columns.stride(1);

    let out_x = ABSOLUTE_POS % out_w;
    let out_y = ABSOLUTE_POS / out_w % out_h;
    let kernel_x = ABSOLUTE_POS / (out_w * out_h) % kernel_width;
    let kernel_y = ABSOLUTE_POS / (out_w * out_h * kernel_width) % kernel_height;
    let channel = ABSOLUTE_POS / col_stride_channel % channels;
    let batch = ABSOLUTE_POS / col_stride_batch % batch_size;

    let channels_per_weight_group = channels / args.weight_groups;

    let weight_group = (channels + channel) % args.weight_groups;
    let in_channels_start = channels_per_weight_group * weight_group;
    let in_channels_end = in_channels_start + channels_per_weight_group;

    let channels_per_offset_group = channels / args.offset_groups;
    let offset_group = channel / channels_per_offset_group;

    let offset_stride_batch = offset.stride(0);
    let offset_stride_index = offset.stride(1);
    let offset_stride_y = offset.stride(2);
    let offset_stride_x = offset.stride(3);

    let mask_stride_batch = mask.stride(0);
    let mask_stride_index = mask.stride(1);
    let mask_stride_y = mask.stride(2);
    let mask_stride_x = mask.stride(3);

    let offset_base_idx =
        batch * offset_stride_batch + offset_group * (offset.stride(1) / args.offset_groups);
    let mask_base_idx =
        batch * mask_stride_batch + offset_group * (mask.stride(1) / args.offset_groups);

    let mask_index = kernel_y * kernel_width + kernel_x;
    let offset_index = UInt::new(2) * mask_index;

    let offset_idx = offset_base_idx + out_y * offset_stride_y + out_x * offset_stride_x;
    let offset_x = offset[offset_idx + offset_index * offset_stride_index];
    let offset_y = offset[offset_idx + (offset_index + 1) * offset_stride_index];

    let mask_value = mask[mask_base_idx
        + mask_index * mask_stride_index
        + out_y * mask_stride_y
        + out_x * mask_stride_x];

    let y =
        F::cast_from((out_y * args.conv_stride_h - args.padding_h) + kernel_y * args.dilation_h)
            + offset_y;
    let x =
        F::cast_from((out_x * args.conv_stride_w - args.padding_w) + kernel_x * args.dilation_w)
            + offset_x;

    let one = UInt::new(1);
    let f_one = F::new(1.0);

    for in_channel in range(in_channels_start, in_channels_end, Comptime::new(false)) {
        for delta_y in range(0, 3, Comptime::new(true)) {
            for delta_x in range(0, 3, Comptime::new(true)) {
                let y_point = UInt::cast_from(y) + delta_y;
                let x_point = UInt::cast_from(x) + delta_x;

                if y_point >= one
                    && y_point <= args.height
                    && x_point > one
                    && x_point <= args.width
                    && F::abs(y - F::cast_from(y_point) - f_one) < f_one
                    && F::abs(x - F::cast_from(x_point) - f_one) < f_one
                {
                    let y_point_2 = y_point - one;
                    let x_point_2 = x_point - one;

                    let gradient_position = batch * input_gradient.stride(0)
                        + in_channel * input_gradient.stride(1)
                        + y_point_2 * input_gradient.stride(2)
                        + x_point_2 * input_gradient.stride(3);
                    let weight = (f_one - F::abs(y - F::cast_from(y_point_2)))
                        * (f_one - F::abs(x - F::abs(x - F::cast_from(x_point_2))));
                    input_gradient[gradient_position] = mask_value * weight * columns[ABSOLUTE_POS];
                }
            }
        }
    }
}
