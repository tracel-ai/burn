use alloc::vec;
use alloc::vec::Vec;

use crate::tensor::FloatTensor;
use crate::{Backend, TensorMetadata};
use burn_std::{MatmulTransformAction, MatmulTransformAnalysis, MatmulTransformPolicy, Shape};

/// Default [linear](crate::ops::ModuleOps::linear) forward implementation.
///
/// Computes `y = x @ weight [+ bias]`.
///
/// The weight is shared by every leading dim of x, so when the batched matmul
/// would tile poorly, the [transform policy](MatmulTransformPolicy) folds the
/// leading dims into the rows and the product runs as one `[rows, d_input] @
/// [d_input, d_output]` matmul. Otherwise weight and bias broadcast to x's rank
/// for a batched matmul.
pub(crate) fn linear<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: Option<FloatTensor<B>>,
) -> FloatTensor<B> {
    let shape = x.shape();
    let ndims = shape.num_dims();

    let analysis = MatmulTransformAnalysis::from_shapes(&shape, &weight.shape());

    match MatmulTransformPolicy::default().action(&analysis) {
        MatmulTransformAction::MergeBatches { rows } => {
            let d_input = shape[ndims - 1];
            let d_output = weight.shape()[1];

            let x = B::float_reshape(x, Shape::new([rows, d_input]));
            let mut output = B::float_matmul(x, weight);

            if let Some(bias) = bias {
                let bias = B::float_reshape(bias, Shape::new([1, d_output]));
                output = B::float_add(output, bias);
            }

            let out_dims: Vec<usize> = (0..ndims - 1).map(|i| shape[i]).chain([d_output]).collect();
            B::float_reshape(output, Shape::from(out_dims))
        }
        MatmulTransformAction::Keep => {
            // Reshape weight [d_input, d_output] -> [1, ..., 1, d_input, d_output]
            // for batch matmul.
            let weight = unsqueeze_leading::<B>(weight, ndims);
            let output = B::float_matmul(x, weight);

            match bias {
                Some(bias) => {
                    // Reshape bias [d_output] -> [1, ..., 1, d_output] to match
                    // output rank.
                    let bias = unsqueeze_leading::<B>(bias, ndims);
                    B::float_add(output, bias)
                }
                None => output,
            }
        }
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

/// Reshape `tensor` `[..., d_last]` to `[num_rows, d_last]`, flattening every
/// leading dim into one.
fn flatten_leading<B: Backend>(tensor: FloatTensor<B>) -> FloatTensor<B> {
    let shape = tensor.shape();
    let ndims = shape.num_dims();
    if ndims == 2 {
        return tensor;
    }
    let d_last = shape[ndims - 1];
    let num_rows = shape.num_elements() / d_last;
    B::float_reshape(tensor, Shape::new([num_rows, d_last]))
}

/// Default [linear_x_backward](crate::ops::ModuleOps::linear_x_backward) implementation.
///
/// Computes `dx = output_grad @ weight^T` — a [linear] forward against the
/// transposed weight, so it shares the transform policy.
pub(crate) fn linear_x_backward<B: Backend>(
    weight: FloatTensor<B>,
    output_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    // weight is [d_input, d_output], transpose to [d_output, d_input]
    let weight = B::float_swap_dims(weight, 0, 1);
    linear::<B>(output_grad, weight, None)
}

/// Default [linear_weight_backward](crate::ops::ModuleOps::linear_weight_backward) implementation.
///
/// Computes `dW = x^T @ output_grad` summed over batch dimensions, as a single
/// general matmul on the flattened operands: `[d_input, num_rows] @ [num_rows,
/// d_output]` — the flattening performs the batch reduction.
pub(crate) fn linear_weight_backward<B: Backend>(
    x: FloatTensor<B>,
    output_grad: FloatTensor<B>,
) -> FloatTensor<B> {
    let x = B::float_swap_dims(flatten_leading::<B>(x), 0, 1);
    let output_grad = flatten_leading::<B>(output_grad);
    B::float_matmul(x, output_grad)
}

/// Default [linear_bias_backward](crate::ops::ModuleOps::linear_bias_backward) implementation.
///
/// Computes `db = sum(output_grad)` over all dimensions except the last, as a
/// single reduction on the flattened gradient.
pub(crate) fn linear_bias_backward<B: Backend>(output_grad: FloatTensor<B>) -> FloatTensor<B> {
    let shape = output_grad.shape();
    let d_output = shape[shape.num_dims() - 1];

    let grad = flatten_leading::<B>(output_grad);
    // float_sum_dim preserves rank (keepdim): [num_rows, d_output] -> [1, d_output].
    let grad = B::float_sum_dim(grad, 0);
    B::float_reshape(grad, Shape::new([d_output]))
}
