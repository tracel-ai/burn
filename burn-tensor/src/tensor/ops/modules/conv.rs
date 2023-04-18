use super::{Conv1dBackward, Conv2dBackward};
use crate::{backend::Backend, Shape};
use libm::ceilf;

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
    let padding = ceilf(padding / 2.);

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

    let out = 1 + libm::ceil(
        (size_in + 2 * padding - dilation * (kernel_size - 1) - 1) as f64 / stride as f64,
    ) as usize;
    i64::max(0, out as i64 - size_out as i64) as usize
}

/// Calculate the [1D convolution](crate::ops::ModuleOps::conv1d) backward pass using convolutions.
pub(crate) fn conv1d_backward<B: Backend>(
    x: B::TensorPrimitive<3>,
    weight: B::TensorPrimitive<3>,
    bias: Option<B::TensorPrimitive<1>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    output_grad: B::TensorPrimitive<3>,
) -> Conv1dBackward<B> {
    let [batch_size, channels_in, length_in] = B::shape(&x).dims;
    let [_batch_size, channels_out, length_out] = B::shape(&output_grad).dims;
    let [_, _, kernel_size] = B::shape(&weight).dims;

    let padding_out = calculate_padding_out(
        kernel_size,
        stride,
        padding,
        dilation,
        length_in,
        length_out,
    );

    let x_grad = B::conv_transpose1d(
        output_grad.clone(),
        weight,
        None,
        stride,
        padding,
        padding_out,
        dilation,
    );

    let x_swapped = B::swap_dims(x, 0, 1);
    let output_grad_swapped = B::swap_dims(output_grad, 0, 1);
    let weight_grad_swapped = B::conv1d(
        x_swapped,
        output_grad_swapped.clone(),
        None,
        dilation,
        padding,
        stride,
    );
    let mut weight_grad = B::swap_dims(weight_grad_swapped, 0, 1);

    if B::shape(&weight_grad) != Shape::new([channels_out, channels_in, kernel_size]) {
        weight_grad = B::index(
            weight_grad,
            [0..channels_out, 0..channels_in, 0..kernel_size],
        );
    }

    Conv1dBackward::new(
        x_grad,
        weight_grad,
        bias.map(|b| {
            let grad = output_grad_swapped;
            let grad = B::reshape(grad, Shape::new([channels_out, batch_size * length_out]));
            let grad = B::sum_dim(grad, 1);

            B::reshape(grad, B::shape(&b))
        }),
    )
}

/// Calculate the [2D convolution](crate::ops::ModuleOps::conv2d) backward pass using convolutions.
pub(crate) fn conv2d_backward<B: Backend>(
    x: B::TensorPrimitive<4>,
    weight: B::TensorPrimitive<4>,
    bias: Option<B::TensorPrimitive<1>>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    output_grad: B::TensorPrimitive<4>,
) -> Conv2dBackward<B> {
    let [batch_size, channels_in, height_in, width_in] = B::shape(&x).dims;
    let [_batch_size, channels_out, height_out, width_out] = B::shape(&output_grad).dims;
    let [_, _, kernel_size_1, kernel_size_2] = B::shape(&weight).dims;

    let padding_1_out = calculate_padding_out(
        kernel_size_1,
        stride[0],
        padding[0],
        dilation[0],
        height_in,
        height_out,
    );
    let padding_2_out = calculate_padding_out(
        kernel_size_2,
        stride[1],
        padding[1],
        dilation[1],
        width_in,
        width_out,
    );

    let x_grad = B::conv_transpose2d(
        output_grad.clone(),
        weight,
        None,
        stride,
        padding,
        [padding_1_out, padding_2_out],
        dilation,
    );

    let x_swapped = B::swap_dims(x, 0, 1);
    let output_grad_swapped = B::swap_dims(output_grad, 0, 1);
    let weight_grad_swapped = B::conv2d(
        x_swapped,
        output_grad_swapped.clone(),
        None,
        dilation,
        padding,
        stride,
    );
    let mut weight_grad = B::swap_dims(weight_grad_swapped, 0, 1);

    if B::shape(&weight_grad)
        != Shape::new([channels_out, channels_in, kernel_size_1, kernel_size_2])
    {
        weight_grad = B::index(
            weight_grad,
            [
                0..channels_out,
                0..channels_in,
                0..kernel_size_1,
                0..kernel_size_2,
            ],
        );
    }

    Conv2dBackward::new(
        x_grad,
        weight_grad,
        bias.map(|b| {
            let grad = output_grad_swapped;
            let grad = B::reshape(
                grad,
                Shape::new([channels_out, batch_size * height_out * width_out]),
            );
            let grad = B::sum_dim(grad, 1);

            B::reshape(grad, B::shape(&b))
        }),
    )
}

/// Execute a 1D convolution using a 2D convolution.
pub(crate) fn conv1d_from_conv2d<B: Backend>(
    x: B::TensorPrimitive<3>,
    weight: B::TensorPrimitive<3>,
    bias: Option<B::TensorPrimitive<1>>,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> B::TensorPrimitive<3> {
    let [channels_out, _channels_in, kernel_size] = B::shape(&weight).dims;
    let [batch_size, channels_in, length_in] = B::shape(&x).dims;

    let weight = B::reshape(
        weight,
        Shape::new([channels_out, channels_in, kernel_size, 1]),
    );
    let x = B::reshape(x, Shape::new([batch_size, channels_in, length_in, 1]));

    let tensor = B::conv2d(x, weight, bias, [stride, 1], [padding, 0], [dilation, 1]);
    let [batch_size, channels_out, height_out, _weight_out] = B::shape(&tensor).dims;
    B::reshape(tensor, Shape::from([batch_size, channels_out, height_out]))
}

/// Execute a 1D transposed convolution using a 2D transposed convolution.
pub(crate) fn conv_transpose1d_from_conv_transpose2d<B: Backend>(
    x: B::TensorPrimitive<3>,
    weight: B::TensorPrimitive<3>,
    bias: Option<B::TensorPrimitive<1>>,
    stride: usize,
    padding: usize,
    padding_out: usize,
    dilation: usize,
) -> B::TensorPrimitive<3> {
    let [channels_in, channels_out, kernel_size] = B::shape(&weight).dims;
    let [batch_size, _channels_in, length_in] = B::shape(&x).dims;

    let weight = B::reshape(
        weight,
        Shape::new([channels_in, channels_out, kernel_size, 1]),
    );
    let x = B::reshape(x, Shape::new([batch_size, channels_in, length_in, 1]));

    let tensor = B::conv_transpose2d(
        x,
        weight,
        bias,
        [stride, 1],
        [padding, 0],
        [padding_out, 0],
        [dilation, 1],
    );
    let [batch_size, channels_out, height_out, _weight_out] = B::shape(&tensor).dims;
    B::reshape(tensor, Shape::from([batch_size, channels_out, height_out]))
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
