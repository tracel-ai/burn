use crate::{
    Shape, TensorMetadata,
    backend::Backend,
    ops::{FloatTensor, IntTensor},
};

use super::{MaxPool1dBackward, MaxPool1dWithIndices};

pub(crate) fn avg_pool1d_from_2d<B: Backend>(
    x: FloatTensor<B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
) -> FloatTensor<B> {
    let [batch_size, channels, length] = x.shape().dims();

    let x = B::float_reshape(x, Shape::from([batch_size, channels, length, 1]));
    let x = B::avg_pool2d(
        x,
        [kernel_size, 1],
        [stride, 1],
        [padding, 0],
        count_include_pad,
    );

    let [batch_size, channels, length, _] = x.shape().dims();

    B::float_reshape(x, Shape::from([batch_size, channels, length]))
}

pub(crate) fn avg_pool1d_backward_from_2d<B: Backend>(
    x: FloatTensor<B>,
    grad: FloatTensor<B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
) -> FloatTensor<B> {
    let [batch_size, channels, length_in] = x.shape().dims();
    let [_, _, length_out] = grad.shape().dims();

    let x = B::float_reshape(x, Shape::from([batch_size, channels, length_in, 1]));
    let grad_x = B::float_reshape(grad, Shape::from([batch_size, channels, length_out, 1]));

    let grad_x = B::avg_pool2d_backward(
        x,
        grad_x,
        [kernel_size, 1],
        [stride, 1],
        [padding, 0],
        count_include_pad,
    );

    B::float_reshape(grad_x, Shape::from([batch_size, channels, length_in]))
}

pub(crate) fn adaptive_avg_pool1d_from_2d<B: Backend>(
    x: FloatTensor<B>,
    output_size: usize,
) -> FloatTensor<B> {
    let [batch_size, channels, length] = x.shape().dims();

    let x = B::float_reshape(x, Shape::from([batch_size, channels, length, 1]));
    let x = B::adaptive_avg_pool2d(x, [output_size, 1]);

    let [batch_size, channels, length, _] = x.shape().dims();

    B::float_reshape(x, Shape::from([batch_size, channels, length]))
}

pub(crate) fn adaptive_avg_pool1d_backward_from_2d<B: Backend>(
    x: FloatTensor<B>,
    grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let [batch_size, channels, length_in] = x.shape().dims();
    let [_, _, length_out] = grad.shape().dims();

    let x = B::float_reshape(x, Shape::from([batch_size, channels, length_in, 1]));
    let grad_x = B::float_reshape(grad, Shape::from([batch_size, channels, length_out, 1]));

    let grad_x = B::adaptive_avg_pool2d_backward(x, grad_x);

    B::float_reshape(grad_x, Shape::from([batch_size, channels, length_in]))
}

pub(crate) fn max_pool1d_from_2d<B: Backend>(
    x: FloatTensor<B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> FloatTensor<B> {
    let [batch_size, channels, length] = x.shape().dims();

    let x = B::float_reshape(x, Shape::from([batch_size, channels, length, 1]));
    let x = B::max_pool2d(
        x,
        [kernel_size, 1],
        [stride, 1],
        [padding, 0],
        [dilation, 1],
    );

    let [batch_size, channels, length, _] = x.shape().dims();

    B::float_reshape(x, Shape::from([batch_size, channels, length]))
}

pub(crate) fn max_pool1d_with_indices_from_2d<B: Backend>(
    x: FloatTensor<B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> MaxPool1dWithIndices<B> {
    let [batch_size, channels, length] = x.shape().dims();

    let x = B::float_reshape(x, Shape::from([batch_size, channels, 1, length]));
    let x = B::max_pool2d_with_indices(
        x,
        [1, kernel_size],
        [1, stride],
        [0, padding],
        [1, dilation],
    );
    let [batch_size, channels, _, length] = x.output.shape().dims();
    let output = B::float_reshape(x.output, Shape::from([batch_size, channels, length]));
    let indices = B::int_reshape(x.indices, Shape::from([batch_size, channels, length]));
    MaxPool1dWithIndices::new(output, indices)
}

pub(crate) fn max_pool1d_with_indices_backward_from_2d<B: Backend>(
    x: FloatTensor<B>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    output_grad: FloatTensor<B>,
    indices: IntTensor<B>,
) -> MaxPool1dBackward<B> {
    let [batch_size, channels, length_in] = x.shape().dims();
    let [_, _, length_out] = output_grad.shape().dims();

    let x = B::float_reshape(x, Shape::from([batch_size, channels, length_in, 1]));
    let grad_x = B::float_reshape(
        output_grad,
        Shape::from([batch_size, channels, length_out, 1]),
    );
    let indices = B::int_reshape(indices, Shape::from([batch_size, channels, length_out, 1]));

    let grad_x = B::max_pool2d_with_indices_backward(
        x,
        [kernel_size, 1],
        [stride, 1],
        [padding, 0],
        [dilation, 1],
        grad_x,
        indices,
    )
    .x_grad;

    MaxPool1dBackward::new(B::float_reshape(
        grad_x,
        Shape::from([batch_size, channels, length_in]),
    ))
}
