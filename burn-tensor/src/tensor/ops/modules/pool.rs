use crate::{backend::Backend, Shape};

pub(crate) fn avg_pool1d_from_avg_pool2d<B: Backend>(
    x: B::TensorPrimitive<3>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> B::TensorPrimitive<3> {
    let [batch_size, channels, length] = B::shape(&x).dims;

    let x = B::reshape(x, Shape::from([batch_size, channels, length, 1]));
    let x = B::avg_pool2d(x, [kernel_size, 1], [stride, 1], [padding, 0]);

    let [batch_size, channels, length, _] = B::shape(&x).dims;
    let x = B::reshape(x, Shape::from([batch_size, channels, length]));

    x
}

pub(crate) fn avg_pool1d_backward_from_avg_pool2d<B: Backend>(
    x: B::TensorPrimitive<3>,
    grad: B::TensorPrimitive<3>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> B::TensorPrimitive<3> {
    let [batch_size, channels, length] = B::shape(&x).dims;

    let x = B::reshape(x, Shape::from([batch_size, channels, length, 1]));
    let grad = B::reshape(grad, Shape::from([batch_size, channels, length, 1]));
    let grad = B::avg_pool2d_backward(x, grad, [kernel_size, 1], [stride, 1], [padding, 0]);

    let [batch_size, channels, length, _] = B::shape(&grad).dims;
    let grad = B::reshape(grad, Shape::from([batch_size, channels, length]));

    grad
}
