use alloc::vec;

use crate::tensor::FloatTensor;
use crate::{Backend, TensorMetadata};
use burn_std::Shape;

/// Default [linear](crate::ops::ModuleOps::linear) forward implementation.
///
/// Computes `y = x @ weight [+ bias]`.
///
/// Weight `[d_input, d_output]` and bias `[d_output]` are broadcast to match x's rank
/// before the matmul.
pub(crate) fn linear<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: Option<FloatTensor<B>>,
) -> FloatTensor<B> {
    let x_ndims = x.shape().num_dims();

    // Reshape weight [d_input, d_output] -> [1, ..., 1, d_input, d_output] for batch matmul.
    let weight = unsqueeze_leading::<B>(weight, x_ndims);
    let output = B::float_matmul(x, weight);

    match bias {
        Some(bias) => {
            // Reshape bias [d_output] -> [1, ..., 1, d_output] to match output rank.
            let bias = unsqueeze_leading::<B>(bias, x_ndims);
            B::float_add(output, bias)
        }
        None => output,
    }
}

/// Reshape a tensor by prepending size-1 dimensions until it has `target_ndims` dimensions.
fn unsqueeze_leading<B: Backend>(tensor: FloatTensor<B>, target_ndims: usize) -> FloatTensor<B> {
    let shape = tensor.shape();
    let ndims = shape.num_dims();
    if ndims >= target_ndims {
        return tensor;
    }
    let mut new_dims = vec![1usize; target_ndims - ndims];
    for i in 0..ndims {
        new_dims.push(shape[i]);
    }
    B::float_reshape(tensor, Shape::from(new_dims))
}

/// Default [linear_x_backward](crate::ops::ModuleOps::linear_x_backward) implementation.
///
/// Computes `dx = output_grad @ weight^T`.
pub(crate) fn linear_x_backward<B: Backend>(
    weight: FloatTensor<B>,
    output_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    // weight is [d_input, d_output], transpose to [d_output, d_input]
    let weight = B::float_swap_dims(weight, 0, 1);
    // Unsqueeze to match output_grad rank for batch matmul
    let grad_ndims = output_grad.shape().num_dims();
    let weight = unsqueeze_leading::<B>(weight, grad_ndims);
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
    // float_sum_dim preserves rank (keepdim), so sum each batch dim at its index.
    let ndims = grad.shape().num_dims();
    if ndims > 2 {
        for dim in 0..ndims - 2 {
            grad = B::float_sum_dim(grad, dim);
        }
        let shape = grad.shape();
        let d0 = shape[ndims - 2];
        let d1 = shape[ndims - 1];
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
    // float_sum_dim preserves rank (keepdim), so sum each dim at its index.
    for dim in 0..ndims - 1 {
        grad = B::float_sum_dim(grad, dim);
    }

    let shape = grad.shape();
    let d_output = shape[ndims - 1];
    B::float_reshape(grad, Shape::new([d_output]))
}
