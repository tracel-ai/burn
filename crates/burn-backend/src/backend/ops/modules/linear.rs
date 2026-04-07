use crate::tensor::FloatTensor;
use crate::{Backend, TensorMetadata};
use burn_std::Shape;

/// Default [linear](crate::ops::ModuleOps::linear) forward implementation.
///
/// Computes `y = x @ weight [+ bias]`.
pub(crate) fn linear<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: Option<FloatTensor<B>>,
) -> FloatTensor<B> {
    let output = B::float_matmul(x, weight);
    match bias {
        Some(bias) => B::float_add(output, bias),
        None => output,
    }
}

/// Default [linear_x_backward](crate::ops::ModuleOps::linear_x_backward) implementation.
///
/// Computes `dx = output_grad @ weight^T`.
pub(crate) fn linear_x_backward<B: Backend>(
    weight: FloatTensor<B>,
    output_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let ndims = weight.shape().num_dims();
    let weight = B::float_swap_dims(weight, ndims - 2, ndims - 1);
    B::float_matmul(output_grad, weight)
}

/// Default [linear_weight_backward](crate::ops::ModuleOps::linear_weight_backward) implementation.
///
/// Computes `dW = x^T @ output_grad`, summed over batch dimensions.
pub(crate) fn linear_weight_backward<B: Backend>(
    x: FloatTensor<B>,
    output_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let ndims = x.shape().num_dims();
    let x = B::float_swap_dims(x, ndims - 2, ndims - 1);
    let mut grad = B::float_matmul(x, output_grad);

    // Sum over all batch dimensions (all dims except the last two).
    let ndims = grad.shape().num_dims();
    if ndims > 2 {
        // Sum from the outermost batch dim inward, always summing dim 0
        // since the tensor shrinks each iteration.
        for _ in 0..ndims - 2 {
            grad = B::float_sum_dim(grad, 0);
        }
        let shape = grad.shape();
        let d0 = shape[shape.num_dims() - 2];
        let d1 = shape[shape.num_dims() - 1];
        B::float_reshape(grad, Shape::new([d0, d1]))
    } else {
        grad
    }
}

/// Default [linear_bias_backward](crate::ops::ModuleOps::linear_bias_backward) implementation.
///
/// Computes `db = sum(output_grad)` over all dimensions except the last.
pub(crate) fn linear_bias_backward<B: Backend>(output_grad: FloatTensor<B>) -> FloatTensor<B> {
    let ndims = output_grad.shape().num_dims();
    let mut grad = output_grad;

    // Sum over all dims except the last (the output feature dim).
    // Sum from the outermost dim inward, always summing dim 0.
    for _ in 0..ndims - 1 {
        grad = B::float_sum_dim(grad, 0);
    }

    let shape = grad.shape();
    let d_output = shape[shape.num_dims() - 1];
    B::float_reshape(grad, Shape::new([d_output]))
}
