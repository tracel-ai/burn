use cubecl::{calculate_cube_count_elemwise, prelude::*};

use burn_tensor::{
    ops::{conv::calculate_conv_output_size, DeformConvOptions},
    Shape,
};

use crate::{
    kernel::into_contiguous,
    ops::{
        numeric::{empty_device, ones_device, zeros_device},
        reshape,
    },
    tensor::JitTensor,
    FloatElement, JitRuntime,
};

#[derive(CubeLaunch)]
struct DeformConv2dArgs {
    conv_stride_h: UInt,
    conv_stride_w: UInt,
    dilation_h: UInt,
    dilation_w: UInt,
    padding_h: UInt,
    padding_w: UInt,
    weight_groups: UInt,
    offset_groups: UInt,
}

#[cube(launch)]
fn deform_conv2d_kernel<F: Float>(
    input: &Tensor<F>,
    offset: &Tensor<F>,
    weight: &Tensor<F>,
    mask: &Tensor<F>,
    bias: &Tensor<F>,
    output: &mut Tensor<F>,
    args: &DeformConv2dArgs,
    kernel_size_0_unroll: Comptime<Option<UInt>>,
    kernel_size_1_unroll: Comptime<Option<UInt>>,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }

    let in_channels = weight.shape(1);

    let kernel_height = Comptime::unwrap_or_else(kernel_size_0_unroll, || weight.shape(2));
    let unroll_y = Comptime::is_some(kernel_size_0_unroll);
    let kernel_width = Comptime::unwrap_or_else(kernel_size_1_unroll, || weight.shape(3));
    let unroll_x = Comptime::is_some(kernel_size_1_unroll);

    let out_batch = ABSOLUTE_POS / output.stride(0) % output.shape(0);
    let out_channel = ABSOLUTE_POS / output.stride(1) % output.shape(1);
    let out_y = ABSOLUTE_POS / output.stride(2) % output.shape(2);
    let out_x = ABSOLUTE_POS / output.stride(3) % output.shape(3);

    let weight_group = (weight.shape(0) + out_channel) % args.weight_groups;
    let offset_group = (offset.shape(0) + out_batch) % args.offset_groups;
    let in_channels_start = in_channels * weight_group;
    let in_channels_end = in_channels_start + in_channels;
    let mut sum = bias[out_channel];

    let index_y_base = out_y * args.conv_stride_h;
    let index_x_base = out_x * args.conv_stride_w;

    let weight_stride_in_channel = weight.stride(1);
    let weight_stride_y = weight.stride(2);
    let weight_stride_x = weight.stride(3);

    let mask_stride_index = mask.stride(1);
    let mask_stride_y = mask.stride(2);
    let mask_stride_x = mask.stride(3);

    let offset_stride_index = offset.stride(1);
    let offset_stride_y = offset.stride(2);
    let offset_stride_x = offset.stride(3);

    let input_stride_channel = input.stride(1);
    let input_stride_y = input.stride(2);
    let input_stride_x = input.stride(3);
    let input_height = input.shape(2);
    let input_width = input.shape(3);

    let border_top = args.padding_h;
    let border_left = args.padding_w;
    let border_bottom = input_height + args.padding_h;
    let border_right = input_width + args.padding_w;

    let batch_index = out_batch * input.stride(0);
    let weight_out_channel_index = out_channel * weight.stride(0);
    let offset_batch_index = out_batch * offset.stride(0);
    let mask_batch_index = out_batch * mask.stride(0);

    for in_channel in range(in_channels_start, in_channels_end, Comptime::new(false)) {
        let in_channel_index = in_channel * input_stride_channel;
        let weight_in_channel_index = (in_channel - in_channels_start) * weight_stride_in_channel;

        for kernel_y in range(0, kernel_height, unroll_y) {
            for kernel_x in range(0, kernel_width, unroll_x) {
                let index_y = kernel_y * args.dilation_h + index_y_base;
                let index_x = kernel_x * args.dilation_w + index_x_base;

                let within_padding = index_y >= border_top
                    && index_y < border_bottom
                    && index_x >= border_left
                    && index_x < border_right;

                if within_padding {
                    let index_y_padded = index_y - args.padding_h;
                    let index_x_padded = index_x - args.padding_w;

                    let mask_index =
                        offset_group * args.offset_groups + kernel_y * kernel_width + kernel_x;
                    let index_offset = UInt::new(2) * mask_index;

                    let mask_value = mask[mask_batch_index
                        + mask_index * mask_stride_index
                        + out_y * mask_stride_y
                        + out_x * mask_stride_x];

                    let offset_base_idx =
                        offset_batch_index + out_y * offset_stride_y + out_x * offset_stride_x;
                    let offset_y = offset[offset_base_idx + index_offset * offset_stride_index];
                    let offset_x =
                        offset[offset_base_idx + (index_offset + 1) * offset_stride_index];

                    let y = offset_y + F::cast_from(index_y_padded);
                    let x = offset_x + F::cast_from(index_x_padded);

                    // Bilinear interpolate
                    let interpolated = bilinear_interpolate(
                        input,
                        y,
                        x,
                        F::cast_from(input_width),
                        F::cast_from(input_height),
                        input_stride_x,
                        input_stride_y,
                        batch_index + in_channel_index,
                    );

                    let index_weight = weight_out_channel_index
                        + weight_in_channel_index
                        + kernel_y * weight_stride_y
                        + kernel_x * weight_stride_x;

                    /*                     let index_input = batch_index
                                           + in_channel_index
                                           + index_y_padded * input_stride_y
                                           + index_x_padded * input_stride_x;
                    */
                    sum += interpolated * weight[index_weight] * mask_value;
                }
            }
        }
    }

    output[ABSOLUTE_POS] = sum;
}

#[cube]
fn bilinear_interpolate<F: Float>(
    input: &Tensor<F>,
    y: F,
    x: F,
    width: F,
    height: F,
    stride_x: UInt,
    stride_y: UInt,
    offset: UInt,
) -> F {
    let zero = F::new(0.0);
    let one = F::new(1.0);
    let neg_one = F::new(0.0 - 1.0);

    let mut result = zero;
    if (y <= neg_one || height <= y || x <= neg_one || width <= x) == false {
        let y_low = F::floor(y);
        let x_low = F::floor(x);
        let y_high = y_low + one;
        let x_high = x_low + one;

        let mut v1 = zero;
        let mut v2 = zero;
        let mut v3 = zero;
        let mut v4 = zero;
        if y_low >= zero && x_low >= zero {
            v1 = input
                [offset + UInt::cast_from(y_low) * stride_y + UInt::cast_from(x_low) * stride_x]
        };
        if y_low >= zero && x_high <= width - one {
            v2 = input
                [offset + UInt::cast_from(y_low) * stride_y + UInt::cast_from(x_high) * stride_x]
        };
        if y_high <= height - one && x_low >= zero {
            v3 = input
                [offset + UInt::cast_from(y_high) * stride_y + UInt::cast_from(x_low) * stride_x]
        };
        if y_high <= height - one && x_high <= width - one {
            v4 = input
                [offset + UInt::cast_from(y_high) * stride_y + UInt::cast_from(x_high) * stride_x]
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

pub(crate) fn deform_conv2d<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R, E, 4>,
    offset: JitTensor<R, E, 4>,
    weight: JitTensor<R, E, 4>,
    mask: Option<JitTensor<R, E, 4>>,
    bias: Option<JitTensor<R, E, 1>>,
    options: DeformConvOptions<2>,
) -> JitTensor<R, E, 4> {
    let input = into_contiguous(input);
    let offset = into_contiguous(offset);
    let weight = into_contiguous(weight);
    let mask = mask.map(|it| into_contiguous(it));
    let bias = bias.map(|it| into_contiguous(it));

    let [batch_size, _, in_height, in_width] = input.shape.dims;
    let [out_channels, _, kernel_height, kernel_width] = weight.shape.dims;

    let out_height = calculate_conv_output_size(
        kernel_height,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        in_height,
    );
    let out_width = calculate_conv_output_size(
        kernel_width,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
        in_width,
    );

    let shape_out = Shape::new([batch_size, out_channels, out_height, out_width]);

    let output = empty_device(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
    );

    if batch_size == 0 {
        return output;
    }

    let bias = match bias {
        Some(bias) => {
            let shape = Shape::from([bias.shape.dims[0], 1, 1, 1]);
            reshape(bias, shape)
        }
        None => {
            let shape = Shape::from([output.shape.dims[0], 1, 1, 1]);
            zeros_device(input.client.clone(), input.device.clone(), shape)
        }
    };
    let mask = if let Some(mask) = mask {
        mask
    } else {
        let shape = Shape::from([
            batch_size,
            options.offset_groups * kernel_height * kernel_width,
            out_height,
            out_width,
        ]);
        ones_device(input.client.clone(), input.device.clone(), shape)
    };

    let num_elems_output = output.shape.num_elements();
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(num_elems_output, cube_dim);

    deform_conv2d_kernel::launch::<E::FloatPrimitive, R>(
        &input.client,
        cube_count,
        cube_dim,
        TensorArg::new(&input.handle, &input.strides, &input.shape.dims),
        TensorArg::new(&offset.handle, &offset.strides, &offset.shape.dims),
        TensorArg::new(&weight.handle, &weight.strides, &weight.shape.dims),
        TensorArg::new(&mask.handle, &mask.strides, &mask.shape.dims),
        TensorArg::new(&bias.handle, &bias.strides, &bias.shape.dims),
        TensorArg::new(&output.handle, &output.strides, &output.shape.dims),
        DeformConv2dArgsLaunch::new(
            ScalarArg::new(options.stride[0] as u32),
            ScalarArg::new(options.stride[1] as u32),
            ScalarArg::new(options.dilation[0] as u32),
            ScalarArg::new(options.dilation[1] as u32),
            ScalarArg::new(options.padding[0] as u32),
            ScalarArg::new(options.padding[1] as u32),
            ScalarArg::new(options.weight_groups as u32),
            ScalarArg::new(options.offset_groups as u32),
        ),
        Some(kernel_height.into()),
        Some(kernel_width.into()),
    );

    output
}
