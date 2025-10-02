#![allow(clippy::single_range_in_vec_init)]
use super::{ConvOptions, ConvTransposeOptions};
use crate::{Shape, Slice, TensorMetadata, backend::Backend, ops::FloatTensor};

use alloc::{vec, vec::Vec};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Calculate the expected padding size required when applying a convolution.
pub fn calculate_conv_padding(
    kernel_size: usize,
    stride: usize,
    size_in: usize,
    size_out: usize,
) -> usize {
    let kernel_size = kernel_size as f32;
    let stride = stride as f32;
    let size_in = size_in as f32;
    let size_out = size_out as f32;

    let padding = stride * (size_out - 1.) - size_in + kernel_size;
    let padding = (padding / 2.).ceil();

    padding as usize
}

/// Calculate the expected output size when doing a convolution operation.
pub fn calculate_conv_output_size(
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    size_in: usize,
) -> usize {
    (size_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
}

/// Calculate the expected output sizes when doing a convolution operation.
pub fn calculate_conv_output_sizes(
    kernel_size: &[usize],
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    size_in: &[usize],
) -> Vec<usize> {
    size_in
        .iter()
        .enumerate()
        .map(|(i, size_in)| {
            calculate_conv_output_size(kernel_size[i], stride[i], padding[i], dilation[i], *size_in)
        })
        .collect()
}

/// Calculate the expected output size when doing a transposed convolution operation.
pub fn calculate_conv_transpose_output_size(
    kernel_size: usize,
    stride: usize,
    padding: usize,
    padding_out: usize,
    dilation: usize,
    size_in: usize,
) -> usize {
    (size_in - 1) * stride + (dilation * (kernel_size - 1) + 1) + padding_out - 2 * padding
}

/// Calculate the expected output size when doing a pooling operation.
pub fn calculate_pool_output_size(
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    size_in: usize,
) -> usize {
    ((size_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
}

/// Calculate the [1D convolution](crate::ops::ModuleOps::conv1d) backward pass, returning the gradient for `x`.
pub(crate) fn conv1d_x_backward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvOptions<1>,
) -> FloatTensor<B> {
    let weight_shape = weight.shape();

    let [_batch_size, _, length_in] = x.shape().dims();
    let [_batch_size, _channels_out, length_out] = output_grad.shape().dims();
    let [_, _, kernel_size] = weight_shape.dims();

    let padding_out = calculate_padding_out(
        kernel_size,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        length_in,
        length_out,
    );

    B::conv_transpose1d(
        output_grad,
        weight,
        None,
        ConvTransposeOptions::new(
            options.stride,
            options.padding,
            [padding_out],
            options.dilation,
            options.groups,
        ),
    )
}

/// Calculate the [1D convolution](crate::ops::ModuleOps::conv1d) backward pass, returning the gradient for `weight`.
pub(crate) fn conv1d_weight_backward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvOptions<1>,
) -> FloatTensor<B> {
    let weight_dtype = weight.dtype();
    let weight_shape = weight.shape();
    let weight_device = B::float_device(&weight);

    match options.groups == 1 {
        true => conv1d_weight_grad_no_groups::<B>(x, output_grad, weight_shape, options),
        false => conv1d_weight_grad_groups::<B>(
            x,
            B::float_zeros(weight_shape, &weight_device, weight_dtype.into()),
            output_grad,
            options,
        ),
    }
}

/// Calculate the [1D convolution](crate::ops::ModuleOps::conv1d) backward pass, returning the gradient for `bias`.
pub(crate) fn conv1d_bias_backward<B: Backend>(
    x: FloatTensor<B>,
    bias: FloatTensor<B>,
    output_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let [batch_size, _, _length_in] = x.shape().dims();
    let [_batch_size, channels_out, length_out] = output_grad.shape().dims();

    let grad = B::float_swap_dims(output_grad, 0, 1);
    let grad = B::float_reshape(grad, Shape::new([channels_out, batch_size * length_out]));
    let grad = B::float_sum_dim(grad, 1);

    B::float_reshape(grad, bias.shape())
}

/// Calculate the [2D convolution](crate::ops::ModuleOps::conv2d) backward pass, returning the gradient for `x`.
pub(crate) fn conv2d_x_backward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvOptions<2>,
) -> FloatTensor<B> {
    let weight_shape = weight.shape();

    let [_batch_size, _channels_in, height_in, width_in] = x.shape().dims();
    let [_, _, height_out, width_out] = output_grad.shape().dims();
    let [_channels_out, _, kernel_size_1, kernel_size_2] = weight_shape.dims();

    let padding_1_out = calculate_padding_out(
        kernel_size_1,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        height_in,
        height_out,
    );
    let padding_2_out = calculate_padding_out(
        kernel_size_2,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
        width_in,
        width_out,
    );

    B::conv_transpose2d(
        output_grad,
        weight,
        None,
        ConvTransposeOptions::new(
            options.stride,
            options.padding,
            [padding_1_out, padding_2_out],
            options.dilation,
            options.groups,
        ),
    )
}

/// Calculate the [2D convolution](crate::ops::ModuleOps::conv2d) backward pass, returning the gradient for `weight`.
pub(crate) fn conv2d_weight_backward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvOptions<2>,
) -> FloatTensor<B> {
    let weight_dtype = weight.dtype();
    let weight_shape = weight.shape();
    let weight_device = B::float_device(&weight);

    match options.groups == 1 {
        true => conv2d_weight_grad_no_groups::<B>(x, output_grad, weight_shape, options),
        false => conv2d_weight_grad_groups::<B>(
            x,
            B::float_zeros(weight_shape, &weight_device, weight_dtype.into()),
            output_grad,
            options,
        ),
    }
}

/// Calculate the [2D convolution](crate::ops::ModuleOps::conv2d) backward pass, returning the gradient for `bias`.
pub(crate) fn conv2d_bias_backward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: FloatTensor<B>,
    output_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let weight_shape = weight.shape();

    let [batch_size, _channels_in, _height_in, _width_in] = x.shape().dims();
    let [_, _, height_out, width_out] = output_grad.shape().dims();
    let [channels_out, _, _kernel_size_1, _kernel_size_2] = weight_shape.dims();

    let grad = B::float_swap_dims(output_grad, 0, 1);
    let grad = B::float_reshape(
        grad,
        Shape::new([channels_out, batch_size * height_out * width_out]),
    );
    let grad = B::float_sum_dim(grad, 1);

    B::float_reshape(grad, bias.shape())
}

/// Calculate the [3D convolution](crate::ops::ModuleOps::conv3d) backward pass, returning the gradient for `x`.
pub(crate) fn conv3d_x_backward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvOptions<3>,
) -> FloatTensor<B> {
    let weight_shape = weight.shape();

    let [_batch_size, _channels_in, depth_in, height_in, width_in] = x.shape().dims();
    let [_, _, depth_out, height_out, width_out] = output_grad.shape().dims();
    let [
        _channels_out,
        _,
        kernel_size_1,
        kernel_size_2,
        kernel_size_3,
    ] = weight_shape.dims();

    let padding_1_out = calculate_padding_out(
        kernel_size_1,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        depth_in,
        depth_out,
    );
    let padding_2_out = calculate_padding_out(
        kernel_size_2,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
        height_in,
        height_out,
    );
    let padding_3_out = calculate_padding_out(
        kernel_size_3,
        options.stride[2],
        options.padding[2],
        options.dilation[2],
        width_in,
        width_out,
    );

    B::conv_transpose3d(
        output_grad,
        weight,
        None,
        ConvTransposeOptions::new(
            options.stride,
            options.padding,
            [padding_1_out, padding_2_out, padding_3_out],
            options.dilation,
            options.groups,
        ),
    )
}

/// Calculate the [3D convolution](crate::ops::ModuleOps::conv3d) backward pass, returning the gradient for `weight`.
pub(crate) fn conv3d_weight_backward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvOptions<3>,
) -> FloatTensor<B> {
    let weight_dtype = weight.dtype();
    let weight_shape = weight.shape();
    let weight_device = B::float_device(&weight);

    match options.groups == 1 {
        true => conv3d_weight_grad_no_groups::<B>(x, output_grad, weight_shape, options),
        false => conv3d_weight_grad_groups::<B>(
            x,
            B::float_zeros(weight_shape, &weight_device, weight_dtype.into()),
            output_grad,
            options,
        ),
    }
}

/// Calculate the [3D convolution](crate::ops::ModuleOps::conv3d) backward pass, returning the gradient for `bias`.
pub(crate) fn conv3d_bias_backward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: FloatTensor<B>,
    output_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let weight_shape = weight.shape();

    let [batch_size, _channels_in, _depth_in, _height_in, _width_in] = x.shape().dims();
    let [_, _, depth_out, height_out, width_out] = output_grad.shape().dims();
    let [
        channels_out,
        _,
        _kernel_size_1,
        _kernel_size_2,
        _kernel_size_3,
    ] = weight_shape.dims();

    let grad = B::float_swap_dims(output_grad, 0, 1);
    let grad = B::float_reshape(
        grad,
        Shape::new([
            channels_out,
            batch_size * depth_out * height_out * width_out,
        ]),
    );
    let grad = B::float_sum_dim(grad, 1);

    B::float_reshape(grad, bias.shape())
}

/// Calculate the [1D convolution transpose](crate::ops::ModuleOps::conv_transpose1d) backward pass, returning the gradient for `x`.
pub(crate) fn conv_transpose1d_x_backward<B: Backend>(
    weight: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvTransposeOptions<1>,
) -> FloatTensor<B> {
    B::conv1d(
        output_grad,
        weight,
        None,
        ConvOptions::new(
            options.stride,
            options.padding,
            options.dilation,
            options.groups,
        ),
    )
}

/// Calculate the [1D convolution transpose](crate::ops::ModuleOps::conv_transpose1d) backward pass, returning the gradient for `weight`.
pub(crate) fn conv_transpose1d_weight_backward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvTransposeOptions<1>,
) -> FloatTensor<B> {
    let weight_dtype = weight.dtype();
    let weight_shape = weight.shape();
    let weight_device = B::float_device(&weight);

    match options.groups == 1 {
        true => conv_transpose1d_weight_grad_no_groups::<B>(x, output_grad, weight_shape, options),
        false => conv_transpose1d_weight_grad_groups::<B>(
            x,
            B::float_zeros(weight_shape, &weight_device, weight_dtype.into()),
            output_grad,
            options,
        ),
    }
}

/// Calculate the [1D convolution transpose](crate::ops::ModuleOps::conv_transpose1d) backward pass, returning the gradient for `bias`.
pub(crate) fn conv_transpose1d_bias_backward<B: Backend>(
    x: FloatTensor<B>,
    bias: FloatTensor<B>,
    output_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let [batch_size, _channels_in, _] = x.shape().dims();
    let [_, channels_out, length_out] = output_grad.shape().dims();

    let grad = B::float_swap_dims(output_grad, 0, 1);
    let grad = B::float_reshape(grad, Shape::new([channels_out, batch_size * length_out]));
    let grad = B::float_sum_dim(grad, 1);

    B::float_reshape(grad, bias.shape())
}

/// Calculate the [2D convolution transpose](crate::ops::ModuleOps::conv_transpose2d) backward pass, returning the gradient for `x`.
pub(crate) fn conv_transpose2d_x_backward<B: Backend>(
    weight: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvTransposeOptions<2>,
) -> FloatTensor<B> {
    B::conv2d(
        output_grad,
        weight,
        None,
        ConvOptions::new(
            options.stride,
            options.padding,
            options.dilation,
            options.groups,
        ),
    )
}

/// Calculate the [2D convolution transpose](crate::ops::ModuleOps::conv_transpose2d) backward pass, returning the gradient for `weight`.
pub(crate) fn conv_transpose2d_weight_backward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvTransposeOptions<2>,
) -> FloatTensor<B> {
    let weight_dtype = weight.dtype();
    let weight_shape = weight.shape();
    let weight_device = B::float_device(&weight);

    match options.groups == 1 {
        true => conv_transpose2d_weight_grad_no_groups::<B>(x, output_grad, weight_shape, options),
        false => conv_transpose2d_weight_grad_groups::<B>(
            x,
            B::float_zeros(weight_shape, &weight_device, weight_dtype.into()),
            output_grad,
            options,
        ),
    }
}

/// Calculate the [2D convolution transpose](crate::ops::ModuleOps::conv_transpose2d) backward pass, returning the gradient for `bias`.
pub(crate) fn conv_transpose2d_bias_backward<B: Backend>(
    x: FloatTensor<B>,
    bias: FloatTensor<B>,
    output_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let [batch_size, _channels_in, _, _] = x.shape().dims();
    let [_, channels_out, height_out, width_out] = output_grad.shape().dims();

    let grad = B::float_swap_dims(output_grad, 0, 1);
    let grad = B::float_reshape(
        grad,
        Shape::new([channels_out, batch_size * height_out * width_out]),
    );
    let grad = B::float_sum_dim(grad, 1);

    B::float_reshape(grad, bias.shape())
}

/// Calculate the [3D convolution transpose](crate::ops::ModuleOps::conv_transpose3d) backward pass, returning the gradient for `x`.
pub(crate) fn conv_transpose3d_x_backward<B: Backend>(
    weight: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvTransposeOptions<3>,
) -> FloatTensor<B> {
    B::conv3d(
        output_grad,
        weight,
        None,
        ConvOptions::new(
            options.stride,
            options.padding,
            options.dilation,
            options.groups,
        ),
    )
}

/// Calculate the [3D convolution transpose](crate::ops::ModuleOps::conv_transpose3d) backward pass, returning the gradient for `weight`.
pub(crate) fn conv_transpose3d_weight_backward<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvTransposeOptions<3>,
) -> FloatTensor<B> {
    let weight_dtype = weight.dtype();
    let weight_shape = weight.shape();
    let weight_device = B::float_device(&weight);

    match options.groups == 1 {
        true => conv_transpose3d_weight_grad_no_groups::<B>(x, output_grad, weight_shape, options),
        false => conv_transpose3d_weight_grad_groups::<B>(
            x,
            B::float_zeros(weight_shape, &weight_device, weight_dtype.into()),
            output_grad,
            options,
        ),
    }
}

/// Calculate the [3D convolution transpose](crate::ops::ModuleOps::conv_transpose3d) backward pass, returning the gradient for `bias`.
pub(crate) fn conv_transpose3d_bias_backward<B: Backend>(
    x: FloatTensor<B>,
    bias: FloatTensor<B>,
    output_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let [batch_size, _channels_in, _, _, _] = x.shape().dims();
    let [_, channels_out, depth_out, height_out, width_out] = output_grad.shape().dims();

    let grad = B::float_swap_dims(output_grad, 0, 1);
    let grad = B::float_reshape(
        grad,
        Shape::new([
            channels_out,
            batch_size * depth_out * height_out * width_out,
        ]),
    );
    let grad = B::float_sum_dim(grad, 1);

    B::float_reshape(grad, bias.shape())
}

/// Execute a 1D convolution using a 2D convolution.
pub(crate) fn conv1d_from_conv2d<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: Option<FloatTensor<B>>,
    options: ConvOptions<1>,
) -> FloatTensor<B> {
    let [channels_out, _channels_in, kernel_size] = weight.shape().dims();
    let [batch_size, channels_in, length_in] = x.shape().dims();

    let weight = B::float_reshape(
        weight,
        Shape::new([channels_out, channels_in / options.groups, kernel_size, 1]),
    );
    let x = B::float_reshape(x, Shape::new([batch_size, channels_in, length_in, 1]));

    let tensor = B::conv2d(
        x,
        weight,
        bias,
        ConvOptions::new(
            [options.stride[0], 1],
            [options.padding[0], 0],
            [options.dilation[0], 1],
            options.groups,
        ),
    );
    let [batch_size, channels_out, height_out, _weight_out] = tensor.shape().dims();
    B::float_reshape(tensor, Shape::from([batch_size, channels_out, height_out]))
}

/// Execute a 1D transposed convolution using a 2D transposed convolution.
pub(crate) fn conv_transpose1d_from_conv_transpose2d<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: Option<FloatTensor<B>>,
    options: ConvTransposeOptions<1>,
) -> FloatTensor<B> {
    let [channels_in, channels_out, kernel_size] = weight.shape().dims();
    let [batch_size, _channels_in, length_in] = x.shape().dims();

    let weight = B::float_reshape(
        weight,
        Shape::new([channels_in, channels_out, kernel_size, 1]),
    );
    let x = B::float_reshape(x, Shape::new([batch_size, channels_in, length_in, 1]));

    let tensor = B::conv_transpose2d(
        x,
        weight,
        bias,
        ConvTransposeOptions::new(
            [options.stride[0], 1],
            [options.padding[0], 0],
            [options.padding_out[0], 0],
            [options.dilation[0], 1],
            options.groups,
        ),
    );
    let [batch_size, channels_out, height_out, _weight_out] = tensor.shape().dims();
    B::float_reshape(tensor, Shape::from([batch_size, channels_out, height_out]))
}

fn conv1d_weight_grad_no_groups<B: Backend>(
    x: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    weight_shape: Shape,
    options: ConvOptions<1>,
) -> FloatTensor<B> {
    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);
    let weight_grad_swapped = B::conv1d(
        x_swapped,
        output_grad_swapped,
        None,
        ConvOptions::new(options.dilation, options.padding, options.stride, 1),
    );
    let mut weight_grad = B::float_swap_dims(weight_grad_swapped, 0, 1);

    if weight_grad.shape() != weight_shape {
        let slices = vec![
            Slice::from(0..weight_shape[0]),
            Slice::from(0..weight_shape[1]),
            Slice::from(0..weight_shape[2]),
        ];
        weight_grad = B::float_slice(weight_grad, &slices);
    }
    weight_grad
}

fn conv2d_weight_grad_no_groups<B: Backend>(
    x: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    weight_shape: Shape,
    options: ConvOptions<2>,
) -> FloatTensor<B> {
    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);
    let weight_grad_swapped = B::conv2d(
        x_swapped,
        output_grad_swapped,
        None,
        ConvOptions::new(options.dilation, options.padding, options.stride, 1),
    );
    let mut weight_grad = B::float_swap_dims(weight_grad_swapped, 0, 1);

    if weight_grad.shape() != weight_shape {
        let slices = vec![
            Slice::from(0..weight_shape[0]),
            Slice::from(0..weight_shape[1]),
            Slice::from(0..weight_shape[2]),
            Slice::from(0..weight_shape[3]),
        ];
        weight_grad = B::float_slice(weight_grad, &slices);
    }
    weight_grad
}

fn conv3d_weight_grad_no_groups<B: Backend>(
    x: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    weight_shape: Shape,
    options: ConvOptions<3>,
) -> FloatTensor<B> {
    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);
    let weight_grad_swapped = B::conv3d(
        x_swapped,
        output_grad_swapped,
        None,
        ConvOptions::new(options.dilation, options.padding, options.stride, 1),
    );
    let mut weight_grad = B::float_swap_dims(weight_grad_swapped, 0, 1);

    if weight_grad.shape() != weight_shape {
        let slices = vec![
            Slice::from(0..weight_shape[0]),
            Slice::from(0..weight_shape[1]),
            Slice::from(0..weight_shape[2]),
            Slice::from(0..weight_shape[3]),
            Slice::from(0..weight_shape[4]),
        ];
        weight_grad = B::float_slice(weight_grad, &slices);
    }
    weight_grad
}

fn conv1d_weight_grad_groups<B: Backend>(
    x: FloatTensor<B>,
    mut weight_grad: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvOptions<1>,
) -> FloatTensor<B> {
    let [channels_out, increment_ci, kernel_size] = weight_grad.shape().dims();
    let increment_co = channels_out / options.groups;

    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);

    for g in 0..options.groups {
        let start_idx_ci = g * increment_ci;
        let end_idx_ci = (g + 1) * increment_ci;
        let start_idx_co = g * increment_co;
        let end_idx_co = (g + 1) * increment_co;

        let x_slice = vec![Slice::new(
            start_idx_ci as isize,
            Some(end_idx_ci as isize),
            1,
        )];
        let x = B::float_slice(x_swapped.clone(), &x_slice);
        let grad_slice = vec![Slice::new(
            start_idx_co as isize,
            Some(end_idx_co as isize),
            1,
        )];
        let grad = B::float_slice(output_grad_swapped.clone(), &grad_slice);
        let mut weight_grad_tmp = B::conv1d(
            x,
            grad,
            None,
            ConvOptions::new(options.dilation, options.padding, options.stride, 1),
        );
        weight_grad_tmp = B::float_swap_dims(weight_grad_tmp, 0, 1);
        weight_grad = B::float_slice_assign(
            weight_grad,
            &[
                Slice::from(start_idx_co..end_idx_co),
                Slice::from(0..increment_ci),
                Slice::from(0..kernel_size),
            ],
            weight_grad_tmp,
        );
    }

    weight_grad
}

fn conv2d_weight_grad_groups<B: Backend>(
    x: FloatTensor<B>,
    mut weight_grad: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvOptions<2>,
) -> FloatTensor<B> {
    let [channels_out, increment_ci, kernel_size_1, kernel_size_2] = weight_grad.shape().dims();
    let increment_co = channels_out / options.groups;

    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);

    for g in 0..options.groups {
        let start_idx_ci = g * increment_ci;
        let end_idx_ci = (g + 1) * increment_ci;
        let start_idx_co = g * increment_co;
        let end_idx_co = (g + 1) * increment_co;

        let x_slice = vec![Slice::new(
            start_idx_ci as isize,
            Some(end_idx_ci as isize),
            1,
        )];
        let x = B::float_slice(x_swapped.clone(), &x_slice);
        let grad_slice = vec![Slice::new(
            start_idx_co as isize,
            Some(end_idx_co as isize),
            1,
        )];
        let grad = B::float_slice(output_grad_swapped.clone(), &grad_slice);
        let mut weight_grad_tmp = B::conv2d(
            x,
            grad,
            None,
            ConvOptions::new(options.dilation, options.padding, options.stride, 1),
        );
        weight_grad_tmp = B::float_swap_dims(weight_grad_tmp, 0, 1);
        let [_, _, kernel_size_1_tmp, kernel_size_2_tmp] = weight_grad_tmp.shape().dims();

        if kernel_size_1_tmp != kernel_size_1 || kernel_size_2_tmp != kernel_size_2 {
            let slices = vec![
                Slice::from(0..increment_co),
                Slice::from(0..increment_ci),
                Slice::from(0..kernel_size_1),
                Slice::from(0..kernel_size_2),
            ];
            weight_grad_tmp = B::float_slice(weight_grad_tmp, &slices);
        }

        weight_grad = B::float_slice_assign(
            weight_grad,
            &[
                Slice::from(start_idx_co..end_idx_co),
                Slice::from(0..increment_ci),
                Slice::from(0..kernel_size_1),
                Slice::from(0..kernel_size_2),
            ],
            weight_grad_tmp,
        );
    }

    weight_grad
}

fn conv3d_weight_grad_groups<B: Backend>(
    x: FloatTensor<B>,
    mut weight_grad: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvOptions<3>,
) -> FloatTensor<B> {
    let [
        channels_out,
        increment_ci,
        kernel_size_1,
        kernel_size_2,
        kernel_size_3,
    ] = weight_grad.shape().dims();
    let increment_co = channels_out / options.groups;

    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);

    for g in 0..options.groups {
        let start_idx_ci = g * increment_ci;
        let end_idx_ci = (g + 1) * increment_ci;
        let start_idx_co = g * increment_co;
        let end_idx_co = (g + 1) * increment_co;

        let x_slice = vec![Slice::new(
            start_idx_ci as isize,
            Some(end_idx_ci as isize),
            1,
        )];
        let x = B::float_slice(x_swapped.clone(), &x_slice);
        let grad_slice = vec![Slice::new(
            start_idx_co as isize,
            Some(end_idx_co as isize),
            1,
        )];
        let grad = B::float_slice(output_grad_swapped.clone(), &grad_slice);
        let mut weight_grad_tmp = B::conv3d(
            x,
            grad,
            None,
            ConvOptions::new(options.dilation, options.padding, options.stride, 1),
        );
        weight_grad_tmp = B::float_swap_dims(weight_grad_tmp, 0, 1);
        let [
            _,
            _,
            kernel_size_1_tmp,
            kernel_size_2_tmp,
            kernel_size_3_tmp,
        ] = weight_grad_tmp.shape().dims();

        if kernel_size_1_tmp != kernel_size_1
            || kernel_size_2_tmp != kernel_size_2
            || kernel_size_3_tmp != kernel_size_3
        {
            let slices = vec![
                Slice::from(0..increment_co),
                Slice::from(0..increment_ci),
                Slice::from(0..kernel_size_1),
                Slice::from(0..kernel_size_2),
                Slice::from(0..kernel_size_3),
            ];
            weight_grad_tmp = B::float_slice(weight_grad_tmp, &slices);
        }

        weight_grad = B::float_slice_assign(
            weight_grad,
            &[
                Slice::from(start_idx_co..end_idx_co),
                Slice::from(0..increment_ci),
                Slice::from(0..kernel_size_1),
                Slice::from(0..kernel_size_2),
                Slice::from(0..kernel_size_3),
            ],
            weight_grad_tmp,
        );
    }

    weight_grad
}

fn conv_transpose1d_weight_grad_no_groups<B: Backend>(
    x: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    weight_shape: Shape,
    options: ConvTransposeOptions<1>,
) -> FloatTensor<B> {
    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);
    let weight_grad_swapped = B::conv1d(
        output_grad_swapped,
        x_swapped,
        None,
        ConvOptions::new(options.dilation, options.padding, options.stride, 1),
    );
    let mut weight_grad = B::float_swap_dims(weight_grad_swapped, 0, 1);

    let grad_shape = weight_grad.shape();
    if grad_shape != weight_shape {
        let slices = vec![
            Slice::from(0..weight_shape[0]),
            Slice::from(0..weight_shape[1]),
            Slice::from(0..weight_shape[2]),
        ];
        weight_grad = B::float_slice(weight_grad, &slices);
    }
    weight_grad
}

fn conv_transpose2d_weight_grad_no_groups<B: Backend>(
    x: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    weight_shape: Shape,
    options: ConvTransposeOptions<2>,
) -> FloatTensor<B> {
    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);
    let weight_grad_swapped = B::conv2d(
        output_grad_swapped,
        x_swapped,
        None,
        ConvOptions::new(options.dilation, options.padding, options.stride, 1),
    );
    let mut weight_grad = B::float_swap_dims(weight_grad_swapped, 0, 1);

    let grad_shape = weight_grad.shape();
    if grad_shape != weight_shape {
        let slices = vec![
            Slice::from(0..weight_shape[0]),
            Slice::from(0..weight_shape[1]),
            Slice::from(0..weight_shape[2]),
            Slice::from(0..weight_shape[3]),
        ];
        weight_grad = B::float_slice(weight_grad, &slices);
    }
    weight_grad
}

fn conv_transpose3d_weight_grad_no_groups<B: Backend>(
    x: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    weight_shape: Shape,
    options: ConvTransposeOptions<3>,
) -> FloatTensor<B> {
    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);
    let weight_grad_swapped = B::conv3d(
        output_grad_swapped,
        x_swapped,
        None,
        ConvOptions::new(options.dilation, options.padding, options.stride, 1),
    );
    let mut weight_grad = B::float_swap_dims(weight_grad_swapped, 0, 1);

    let grad_shape = weight_grad.shape();
    if grad_shape != weight_shape {
        let slices = vec![
            Slice::from(0..weight_shape[0]),
            Slice::from(0..weight_shape[1]),
            Slice::from(0..weight_shape[2]),
            Slice::from(0..weight_shape[3]),
            Slice::from(0..weight_shape[4]),
        ];
        weight_grad = B::float_slice(weight_grad, &slices);
    }
    weight_grad
}

fn conv_transpose1d_weight_grad_groups<B: Backend>(
    x: FloatTensor<B>,
    mut weight_grad: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvTransposeOptions<1>,
) -> FloatTensor<B> {
    let [channels_in, increment_co, kernel_size] = weight_grad.shape().dims();
    let increment_ci = channels_in / options.groups;

    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);

    for g in 0..options.groups {
        let start_idx_ci = g * increment_ci;
        let end_idx_ci = (g + 1) * increment_ci;
        let start_idx_co = g * increment_co;
        let end_idx_co = (g + 1) * increment_co;

        let x_slice = vec![Slice::new(
            start_idx_ci as isize,
            Some(end_idx_ci as isize),
            1,
        )];
        let x = B::float_slice(x_swapped.clone(), &x_slice);
        let grad_slice = vec![Slice::new(
            start_idx_co as isize,
            Some(end_idx_co as isize),
            1,
        )];
        let grad = B::float_slice(output_grad_swapped.clone(), &grad_slice);
        let mut weight_grad_tmp = B::conv1d(
            grad,
            x,
            None,
            ConvOptions::new(options.dilation, options.padding, options.stride, 1),
        );
        weight_grad_tmp = B::float_swap_dims(weight_grad_tmp, 0, 1);
        let [_, _, kernel_size_tmp] = weight_grad_tmp.shape().dims();

        if kernel_size_tmp != kernel_size {
            let slices = vec![
                Slice::from(0..increment_ci),
                Slice::from(0..increment_co),
                Slice::from(0..kernel_size),
            ];
            weight_grad_tmp = B::float_slice(weight_grad_tmp, &slices);
        }

        weight_grad = B::float_slice_assign(
            weight_grad,
            &[
                Slice::from(start_idx_ci..end_idx_ci),
                Slice::from(0..increment_co),
                Slice::from(0..kernel_size),
            ],
            weight_grad_tmp,
        );
    }

    weight_grad
}

fn conv_transpose2d_weight_grad_groups<B: Backend>(
    x: FloatTensor<B>,
    mut weight_grad: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvTransposeOptions<2>,
) -> FloatTensor<B> {
    let [channels_in, increment_co, kernel_size_1, kernel_size_2] = weight_grad.shape().dims();
    let increment_ci = channels_in / options.groups;

    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);

    for g in 0..options.groups {
        let start_idx_ci = g * increment_ci;
        let end_idx_ci = (g + 1) * increment_ci;
        let start_idx_co = g * increment_co;
        let end_idx_co = (g + 1) * increment_co;

        let x_slice = vec![Slice::new(
            start_idx_ci as isize,
            Some(end_idx_ci as isize),
            1,
        )];
        let x = B::float_slice(x_swapped.clone(), &x_slice);
        let grad_slice = vec![Slice::new(
            start_idx_co as isize,
            Some(end_idx_co as isize),
            1,
        )];
        let grad = B::float_slice(output_grad_swapped.clone(), &grad_slice);
        let mut weight_grad_tmp = B::conv2d(
            grad,
            x,
            None,
            ConvOptions::new(options.dilation, options.padding, options.stride, 1),
        );
        weight_grad_tmp = B::float_swap_dims(weight_grad_tmp, 0, 1);
        let [_, _, kernel_size_1_tmp, kernel_size_2_tmp] = weight_grad_tmp.shape().dims();

        if kernel_size_1_tmp != kernel_size_1 || kernel_size_2_tmp != kernel_size_2 {
            let slices = vec![
                Slice::from(0..increment_ci),
                Slice::from(0..increment_co),
                Slice::from(0..kernel_size_1),
                Slice::from(0..kernel_size_2),
            ];
            weight_grad_tmp = B::float_slice(weight_grad_tmp, &slices);
        }

        weight_grad = B::float_slice_assign(
            weight_grad,
            &[
                Slice::from(start_idx_ci..end_idx_ci),
                Slice::from(0..increment_co),
                Slice::from(0..kernel_size_1),
                Slice::from(0..kernel_size_2),
            ],
            weight_grad_tmp,
        );
    }

    weight_grad
}

fn conv_transpose3d_weight_grad_groups<B: Backend>(
    x: FloatTensor<B>,
    mut weight_grad: FloatTensor<B>,
    output_grad: FloatTensor<B>,
    options: ConvTransposeOptions<3>,
) -> FloatTensor<B> {
    let [
        channels_in,
        increment_co,
        kernel_size_1,
        kernel_size_2,
        kernel_size_3,
    ] = weight_grad.shape().dims();
    let increment_ci = channels_in / options.groups;

    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);

    for g in 0..options.groups {
        let start_idx_ci = g * increment_ci;
        let end_idx_ci = (g + 1) * increment_ci;
        let start_idx_co = g * increment_co;
        let end_idx_co = (g + 1) * increment_co;

        let x_slice = vec![Slice::new(
            start_idx_ci as isize,
            Some(end_idx_ci as isize),
            1,
        )];
        let x = B::float_slice(x_swapped.clone(), &x_slice);
        let grad_slice = vec![Slice::new(
            start_idx_co as isize,
            Some(end_idx_co as isize),
            1,
        )];
        let grad = B::float_slice(output_grad_swapped.clone(), &grad_slice);
        let mut weight_grad_tmp = B::conv3d(
            grad,
            x,
            None,
            ConvOptions::new(options.dilation, options.padding, options.stride, 1),
        );
        weight_grad_tmp = B::float_swap_dims(weight_grad_tmp, 0, 1);
        let [
            _,
            _,
            kernel_size_1_tmp,
            kernel_size_2_tmp,
            kernel_size_3_tmp,
        ] = weight_grad_tmp.shape().dims();

        if kernel_size_1_tmp != kernel_size_1
            || kernel_size_2_tmp != kernel_size_2
            || kernel_size_3_tmp != kernel_size_3
        {
            let slices = vec![
                Slice::from(0..increment_ci),
                Slice::from(0..increment_co),
                Slice::from(0..kernel_size_1),
                Slice::from(0..kernel_size_2),
                Slice::from(0..kernel_size_3),
            ];
            weight_grad_tmp = B::float_slice(weight_grad_tmp, &slices);
        }
        weight_grad = B::float_slice_assign(
            weight_grad,
            &[
                Slice::from(start_idx_ci..end_idx_ci),
                Slice::from(0..increment_co),
                Slice::from(0..kernel_size_1),
                Slice::from(0..kernel_size_2),
                Slice::from(0..kernel_size_3),
            ],
            weight_grad_tmp,
        );
    }

    weight_grad
}

fn calculate_padding_out(
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    size_in: usize,
    size_out: usize,
) -> usize {
    if stride <= 1 {
        return 0;
    }

    let out = 1
        + ((size_in + 2 * padding - dilation * (kernel_size - 1) - 1) as f64 / stride as f64).ceil()
            as usize;
    i64::max(0, out as i64 - size_out as i64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_output_size_1() {
        let kernel_size = 3;
        let stride = 1;
        let padding = 1;
        let size_in = 3;
        let dilation = 1;

        let size_out = calculate_conv_output_size(kernel_size, stride, padding, dilation, size_in);

        assert_eq!(size_out, 3);
    }

    #[test]
    fn test_calculate_output_size_2() {
        let kernel_size = 5;
        let stride = 2;
        let padding = 3;
        let size_in = 27;
        let dilation = 1;

        let size_out = calculate_conv_output_size(kernel_size, stride, padding, dilation, size_in);

        assert_eq!(size_out, 15);
    }

    #[test]
    fn test_calculate_output_size_3() {
        let kernel_size = 5;
        let stride = 2;
        let padding = 3;
        let size_in = 27;
        let dilation = 2;

        let size_out = calculate_conv_output_size(kernel_size, stride, padding, dilation, size_in);

        assert_eq!(size_out, 13);
    }

    #[test]
    fn test_calculate_same_padding_1() {
        let kernel_size = 3;
        let stride = 1;
        let size_in = 3;
        let dilation = 1;

        let padding = calculate_conv_padding(kernel_size, stride, size_in, size_in);
        let size_out = calculate_conv_output_size(kernel_size, stride, padding, dilation, size_in);

        assert_eq!(size_in, size_out, "Expected size");
    }

    #[test]
    fn test_calculate_same_padding_2() {
        let kernel_size = 3;
        let stride = 2;
        let size_in = 7;
        let dilation = 1;

        let padding = calculate_conv_padding(kernel_size, stride, size_in, size_in);
        let size_out = calculate_conv_output_size(kernel_size, stride, padding, dilation, size_in);

        assert_eq!(size_in, size_out, "Expected size");
    }

    #[test]
    fn test_calculate_output_padding_1() {
        let kernel_size = 3;
        let stride = 2;
        let size_in = 7;
        let size_out = 10;
        let dilation = 1;

        let padding = calculate_conv_padding(kernel_size, stride, size_in, size_out);
        let size_out_expected =
            calculate_conv_output_size(kernel_size, stride, padding, dilation, size_in);

        assert_eq!(size_out, size_out_expected, "Expected size");
    }
}
