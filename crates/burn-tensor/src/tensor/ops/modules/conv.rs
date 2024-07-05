#![allow(clippy::single_range_in_vec_init)]
use super::{Conv1dBackward, Conv2dBackward, ConvOptions, ConvTransposeOptions};
use crate::{backend::Backend, ops::FloatTensor, Shape};

#[cfg(not(feature = "std"))]
use num_traits::Float;

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

/// Calculate the expected output size when doing a transposed convolution operation.
pub fn calculate_conv_transpose_output_size(
    kernel_size: usize,
    stride: usize,
    padding: usize,
    padding_out: usize,
    dilation: usize,
    size_in: usize,
) -> usize {
    (size_in - 1) * stride + dilation * (kernel_size - 1) + padding_out - 2 * padding + 1
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

/// Calculate the [1D convolution](crate::ops::ModuleOps::conv1d) backward pass using convolutions.
pub(crate) fn conv1d_backward<B: Backend>(
    x: FloatTensor<B, 3>,
    weight: FloatTensor<B, 3>,
    bias: Option<FloatTensor<B, 1>>,
    output_grad: FloatTensor<B, 3>,
    options: ConvOptions<1>,
) -> Conv1dBackward<B> {
    let weight_shape = B::float_shape(&weight);
    let weight_device = B::float_device(&weight);

    let [batch_size, _, length_in] = B::float_shape(&x).dims;
    let [_batch_size, channels_out, length_out] = B::float_shape(&output_grad).dims;
    let [_, _, kernel_size] = weight_shape.dims;

    let padding_out = calculate_padding_out(
        kernel_size,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        length_in,
        length_out,
    );

    let x_grad = B::conv_transpose1d(
        output_grad.clone(),
        weight,
        None,
        ConvTransposeOptions::new(
            options.stride,
            options.padding,
            [padding_out],
            options.dilation,
            options.groups,
        ),
    );

    let weight_grad = match options.groups == 1 {
        true => conv1d_weight_grad_no_groups::<B>(x, output_grad.clone(), weight_shape, options),
        false => conv1d_weight_grad_groups::<B>(
            x,
            B::float_zeros(weight_shape, &weight_device),
            output_grad.clone(),
            options,
        ),
    };

    Conv1dBackward::new(
        x_grad,
        weight_grad,
        bias.map(|b| {
            let grad = B::float_swap_dims(output_grad, 0, 1);
            let grad = B::float_reshape(grad, Shape::new([channels_out, batch_size * length_out]));
            let grad = B::float_sum_dim(grad, 1);

            B::float_reshape(grad, B::float_shape(&b))
        }),
    )
}

/// Calculate the [2D convolution](crate::ops::ModuleOps::conv2d) backward pass using convolutions.
pub(crate) fn conv2d_backward<B: Backend>(
    x: FloatTensor<B, 4>,
    weight: FloatTensor<B, 4>,
    bias: Option<FloatTensor<B, 1>>,
    output_grad: FloatTensor<B, 4>,
    options: ConvOptions<2>,
) -> Conv2dBackward<B> {
    let weight_shape = B::float_shape(&weight);
    let weight_device = B::float_device(&weight);

    let [batch_size, _channels_in, height_in, width_in] = B::float_shape(&x).dims;
    let [_, _, height_out, width_out] = B::float_shape(&output_grad).dims;
    let [channels_out, _, kernel_size_1, kernel_size_2] = weight_shape.dims;

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

    let x_grad = B::conv_transpose2d(
        output_grad.clone(),
        weight,
        None,
        ConvTransposeOptions::new(
            options.stride,
            options.padding,
            [padding_1_out, padding_2_out],
            options.dilation,
            options.groups,
        ),
    );

    let weight_grad = match options.groups == 1 {
        true => conv2d_weight_grad_no_groups::<B>(x, output_grad.clone(), weight_shape, options),
        false => conv2d_weight_grad_groups::<B>(
            x,
            B::float_zeros(weight_shape, &weight_device),
            output_grad.clone(),
            options,
        ),
    };

    Conv2dBackward::new(
        x_grad,
        weight_grad,
        bias.map(|b| {
            let grad = B::float_swap_dims(output_grad, 0, 1);
            let grad = B::float_reshape(
                grad,
                Shape::new([channels_out, batch_size * height_out * width_out]),
            );
            let grad = B::float_sum_dim(grad, 1);

            B::float_reshape(grad, B::float_shape(&b))
        }),
    )
}

/// Calculate the [2D convolution transpose](crate::ops::ModuleOps::conv_transpose2d) backward pass using convolutions.
pub(crate) fn conv_transpose2d_backward<B: Backend>(
    x: FloatTensor<B, 4>,
    weight: FloatTensor<B, 4>,
    bias: Option<FloatTensor<B, 1>>,
    output_grad: FloatTensor<B, 4>,
    options: ConvTransposeOptions<2>,
) -> Conv2dBackward<B> {
    let weight_shape = B::float_shape(&weight);
    let weight_device = B::float_device(&weight);

    let [batch_size, _channels_in, _, _] = B::float_shape(&x).dims;
    let [_, channels_out, height_out, width_out] = B::float_shape(&output_grad).dims;

    let x_grad = B::conv2d(
        output_grad.clone(),
        weight,
        None,
        ConvOptions::new(
            options.stride,
            options.padding,
            options.dilation,
            options.groups,
        ),
    );

    let weight_grad = match options.groups == 1 {
        true => conv_transpose2d_weight_grad_no_groups::<B>(
            x,
            output_grad.clone(),
            weight_shape,
            options,
        ),
        false => conv_transpose2d_weight_grad_groups::<B>(
            x,
            B::float_zeros(weight_shape, &weight_device),
            output_grad.clone(),
            options,
        ),
    };

    Conv2dBackward::new(
        x_grad,
        weight_grad,
        bias.map(|b| {
            let grad = B::float_swap_dims(output_grad, 0, 1);
            let grad = B::float_reshape(
                grad,
                Shape::new([channels_out, batch_size * height_out * width_out]),
            );
            let grad = B::float_sum_dim(grad, 1);

            B::float_reshape(grad, B::float_shape(&b))
        }),
    )
}

/// Calculate the [1D convolution transpose](crate::ops::ModuleOps::conv_transpose1d) backward pass using convolutions.
pub(crate) fn conv_transpose1d_backward<B: Backend>(
    x: FloatTensor<B, 3>,
    weight: FloatTensor<B, 3>,
    bias: Option<FloatTensor<B, 1>>,
    output_grad: FloatTensor<B, 3>,
    options: ConvTransposeOptions<1>,
) -> Conv1dBackward<B> {
    let weight_shape = B::float_shape(&weight);
    let weight_device = B::float_device(&weight);

    let [batch_size, _channels_in, _] = B::float_shape(&x).dims;
    let [_, channels_out, length_out] = B::float_shape(&output_grad).dims;

    let x_grad = B::conv1d(
        output_grad.clone(),
        weight,
        None,
        ConvOptions::new(
            options.stride,
            options.padding,
            options.dilation,
            options.groups,
        ),
    );

    let weight_grad = match options.groups == 1 {
        true => conv_transpose1d_weight_grad_no_groups::<B>(
            x,
            output_grad.clone(),
            weight_shape,
            options,
        ),
        false => conv_transpose1d_weight_grad_groups::<B>(
            x,
            B::float_zeros(weight_shape, &weight_device),
            output_grad.clone(),
            options,
        ),
    };

    Conv1dBackward::new(
        x_grad,
        weight_grad,
        bias.map(|b| {
            let grad = B::float_swap_dims(output_grad, 0, 1);
            let grad = B::float_reshape(grad, Shape::new([channels_out, batch_size * length_out]));
            let grad = B::float_sum_dim(grad, 1);

            B::float_reshape(grad, B::float_shape(&b))
        }),
    )
}

/// Execute a 1D convolution using a 2D convolution.
pub(crate) fn conv1d_from_conv2d<B: Backend>(
    x: FloatTensor<B, 3>,
    weight: FloatTensor<B, 3>,
    bias: Option<FloatTensor<B, 1>>,
    options: ConvOptions<1>,
) -> FloatTensor<B, 3> {
    let [channels_out, _channels_in, kernel_size] = B::float_shape(&weight).dims;
    let [batch_size, channels_in, length_in] = B::float_shape(&x).dims;

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
    let [batch_size, channels_out, height_out, _weight_out] = B::float_shape(&tensor).dims;
    B::float_reshape(tensor, Shape::from([batch_size, channels_out, height_out]))
}

/// Execute a 1D transposed convolution using a 2D transposed convolution.
pub(crate) fn conv_transpose1d_from_conv_transpose2d<B: Backend>(
    x: FloatTensor<B, 3>,
    weight: FloatTensor<B, 3>,
    bias: Option<FloatTensor<B, 1>>,
    options: ConvTransposeOptions<1>,
) -> FloatTensor<B, 3> {
    let [channels_in, channels_out, kernel_size] = B::float_shape(&weight).dims;
    let [batch_size, _channels_in, length_in] = B::float_shape(&x).dims;

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
    let [batch_size, channels_out, height_out, _weight_out] = B::float_shape(&tensor).dims;
    B::float_reshape(tensor, Shape::from([batch_size, channels_out, height_out]))
}

fn conv1d_weight_grad_groups<B: Backend>(
    x: FloatTensor<B, 3>,
    mut weight_grad: FloatTensor<B, 3>,
    output_grad: FloatTensor<B, 3>,
    options: ConvOptions<1>,
) -> FloatTensor<B, 3> {
    let [channels_out, increment_ci, kernel_size] = B::float_shape(&weight_grad).dims;
    let increment_co = channels_out / options.groups;

    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);

    for g in 0..options.groups {
        let start_idx_ci = g * increment_ci;
        let end_idx_ci = (g + 1) * increment_ci;
        let start_idx_co = g * increment_co;
        let end_idx_co = (g + 1) * increment_co;

        let x = B::float_slice(x_swapped.clone(), [start_idx_ci..end_idx_ci]);
        let grad = B::float_slice(output_grad_swapped.clone(), [start_idx_co..end_idx_co]);
        let mut weight_grad_tmp = B::conv1d(
            x,
            grad,
            None,
            ConvOptions::new(options.dilation, options.padding, options.stride, 1),
        );
        weight_grad_tmp = B::float_swap_dims(weight_grad_tmp, 0, 1);
        weight_grad = B::float_slice_assign(
            weight_grad,
            [start_idx_co..end_idx_co, 0..increment_ci, 0..kernel_size],
            weight_grad_tmp,
        );
    }

    weight_grad
}

fn conv2d_weight_grad_groups<B: Backend>(
    x: FloatTensor<B, 4>,
    mut weight_grad: FloatTensor<B, 4>,
    output_grad: FloatTensor<B, 4>,
    options: ConvOptions<2>,
) -> FloatTensor<B, 4> {
    let [channels_out, increment_ci, kernel_size_1, kernel_size_2] =
        B::float_shape(&weight_grad).dims;
    let increment_co = channels_out / options.groups;

    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);

    for g in 0..options.groups {
        let start_idx_ci = g * increment_ci;
        let end_idx_ci = (g + 1) * increment_ci;
        let start_idx_co = g * increment_co;
        let end_idx_co = (g + 1) * increment_co;

        let x = B::float_slice(x_swapped.clone(), [start_idx_ci..end_idx_ci]);
        let grad = B::float_slice(output_grad_swapped.clone(), [start_idx_co..end_idx_co]);
        let mut weight_grad_tmp = B::conv2d(
            x,
            grad,
            None,
            ConvOptions::new(options.dilation, options.padding, options.stride, 1),
        );
        weight_grad_tmp = B::float_swap_dims(weight_grad_tmp, 0, 1);
        let [_, _, kernel_size_1_tmp, kernel_size_2_tmp] = B::float_shape(&weight_grad_tmp).dims;

        if kernel_size_1_tmp != kernel_size_1 || kernel_size_2_tmp != kernel_size_2 {
            weight_grad_tmp = B::float_slice(
                weight_grad_tmp,
                [
                    0..increment_ci,
                    0..increment_co,
                    0..kernel_size_1,
                    0..kernel_size_2,
                ],
            );
        }

        weight_grad = B::float_slice_assign(
            weight_grad,
            [
                start_idx_co..end_idx_co,
                0..increment_ci,
                0..kernel_size_1,
                0..kernel_size_2,
            ],
            weight_grad_tmp,
        );
    }

    weight_grad
}

fn conv_transpose2d_weight_grad_groups<B: Backend>(
    x: FloatTensor<B, 4>,
    mut weight_grad: FloatTensor<B, 4>,
    output_grad: FloatTensor<B, 4>,
    options: ConvTransposeOptions<2>,
) -> FloatTensor<B, 4> {
    let [channels_in, increment_co, kernel_size_1, kernel_size_2] =
        B::float_shape(&weight_grad).dims;
    let increment_ci = channels_in / options.groups;

    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);

    for g in 0..options.groups {
        let start_idx_ci = g * increment_ci;
        let end_idx_ci = (g + 1) * increment_ci;
        let start_idx_co = g * increment_co;
        let end_idx_co = (g + 1) * increment_co;

        let x = B::float_slice(x_swapped.clone(), [start_idx_ci..end_idx_ci]);
        let grad = B::float_slice(output_grad_swapped.clone(), [start_idx_co..end_idx_co]);
        let mut weight_grad_tmp = B::conv2d(
            grad,
            x,
            None,
            ConvOptions::new(options.dilation, options.padding, options.stride, 1),
        );
        weight_grad_tmp = B::float_swap_dims(weight_grad_tmp, 0, 1);
        let [_, _, kernel_size_1_tmp, kernel_size_2_tmp] = B::float_shape(&weight_grad_tmp).dims;

        if kernel_size_1_tmp != kernel_size_1 || kernel_size_2_tmp != kernel_size_2 {
            weight_grad_tmp = B::float_slice(
                weight_grad_tmp,
                [
                    0..increment_ci,
                    0..increment_co,
                    0..kernel_size_1,
                    0..kernel_size_2,
                ],
            );
        }

        weight_grad = B::float_slice_assign(
            weight_grad,
            [
                start_idx_ci..end_idx_ci,
                0..increment_co,
                0..kernel_size_1,
                0..kernel_size_2,
            ],
            weight_grad_tmp,
        );
    }

    weight_grad
}

fn conv1d_weight_grad_no_groups<B: Backend>(
    x: FloatTensor<B, 3>,
    output_grad: FloatTensor<B, 3>,
    weight_shape: Shape<3>,
    options: ConvOptions<1>,
) -> FloatTensor<B, 3> {
    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);
    let weight_grad_swapped = B::conv1d(
        x_swapped,
        output_grad_swapped,
        None,
        ConvOptions::new(options.dilation, options.padding, options.stride, 1),
    );
    let mut weight_grad = B::float_swap_dims(weight_grad_swapped, 0, 1);

    if B::float_shape(&weight_grad) != weight_shape {
        weight_grad = B::float_slice(
            weight_grad,
            [
                0..weight_shape.dims[0],
                0..weight_shape.dims[1],
                0..weight_shape.dims[2],
            ],
        );
    }
    weight_grad
}

fn conv_transpose1d_weight_grad_groups<B: Backend>(
    x: FloatTensor<B, 3>,
    mut weight_grad: FloatTensor<B, 3>,
    output_grad: FloatTensor<B, 3>,
    options: ConvTransposeOptions<1>,
) -> FloatTensor<B, 3> {
    let [channels_in, increment_co, kernel_size] = B::float_shape(&weight_grad).dims;
    let increment_ci = channels_in / options.groups;

    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);

    for g in 0..options.groups {
        let start_idx_ci = g * increment_ci;
        let end_idx_ci = (g + 1) * increment_ci;
        let start_idx_co = g * increment_co;
        let end_idx_co = (g + 1) * increment_co;

        let x = B::float_slice(x_swapped.clone(), [start_idx_ci..end_idx_ci]);
        let grad = B::float_slice(output_grad_swapped.clone(), [start_idx_co..end_idx_co]);
        let mut weight_grad_tmp = B::conv1d(
            grad,
            x,
            None,
            ConvOptions::new(options.dilation, options.padding, options.stride, 1),
        );
        weight_grad_tmp = B::float_swap_dims(weight_grad_tmp, 0, 1);
        let [_, _, kernel_size_tmp] = B::float_shape(&weight_grad_tmp).dims;

        if kernel_size_tmp != kernel_size {
            weight_grad_tmp = B::float_slice(
                weight_grad_tmp,
                [0..increment_ci, 0..increment_co, 0..kernel_size],
            );
        }

        weight_grad = B::float_slice_assign(
            weight_grad,
            [start_idx_ci..end_idx_ci, 0..increment_co, 0..kernel_size],
            weight_grad_tmp,
        );
    }

    weight_grad
}

fn conv2d_weight_grad_no_groups<B: Backend>(
    x: FloatTensor<B, 4>,
    output_grad: FloatTensor<B, 4>,
    weight_shape: Shape<4>,
    options: ConvOptions<2>,
) -> FloatTensor<B, 4> {
    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);
    let weight_grad_swapped = B::conv2d(
        x_swapped,
        output_grad_swapped,
        None,
        ConvOptions::new(options.dilation, options.padding, options.stride, 1),
    );
    let mut weight_grad = B::float_swap_dims(weight_grad_swapped, 0, 1);

    if B::float_shape(&weight_grad) != weight_shape {
        weight_grad = B::float_slice(
            weight_grad,
            [
                0..weight_shape.dims[0],
                0..weight_shape.dims[1],
                0..weight_shape.dims[2],
                0..weight_shape.dims[3],
            ],
        );
    }
    weight_grad
}

fn conv_transpose1d_weight_grad_no_groups<B: Backend>(
    x: FloatTensor<B, 3>,
    output_grad: FloatTensor<B, 3>,
    weight_shape: Shape<3>,
    options: ConvTransposeOptions<1>,
) -> FloatTensor<B, 3> {
    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);
    let weight_grad_swapped = B::conv1d(
        output_grad_swapped,
        x_swapped,
        None,
        ConvOptions::new(options.dilation, options.padding, options.stride, 1),
    );
    let mut weight_grad = B::float_swap_dims(weight_grad_swapped, 0, 1);

    let grad_shape = B::float_shape(&weight_grad);

    if grad_shape != weight_shape {
        weight_grad = B::float_slice(
            weight_grad,
            [
                0..weight_shape.dims[0],
                0..weight_shape.dims[1],
                0..weight_shape.dims[2],
            ],
        );
    }
    weight_grad
}

fn conv_transpose2d_weight_grad_no_groups<B: Backend>(
    x: FloatTensor<B, 4>,
    output_grad: FloatTensor<B, 4>,
    weight_shape: Shape<4>,
    options: ConvTransposeOptions<2>,
) -> FloatTensor<B, 4> {
    let x_swapped = B::float_swap_dims(x, 0, 1);
    let output_grad_swapped = B::float_swap_dims(output_grad, 0, 1);
    let weight_grad_swapped = B::conv2d(
        output_grad_swapped,
        x_swapped,
        None,
        ConvOptions::new(options.dilation, options.padding, options.stride, 1),
    );
    let mut weight_grad = B::float_swap_dims(weight_grad_swapped, 0, 1);

    let grad_shape = B::float_shape(&weight_grad);

    if grad_shape != weight_shape {
        weight_grad = B::float_slice(
            weight_grad,
            [
                0..weight_shape.dims[0],
                0..weight_shape.dims[1],
                0..weight_shape.dims[2],
                0..weight_shape.dims[3],
            ],
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
