use alloc::vec;

use crate::tensor::FloatTensor;
use crate::{Backend, TensorMetadata};
use burn_std::Shape;

/// Default [linear](crate::ops::ModuleOps::linear) forward implementation.
///
/// Computes `y = x @ weight [+ bias]`.
///
/// Weight `[d_input, d_output]` and bias `[d_output]` are broadcast to match x's rank
/// before the matmul. The decomposition stays a plain broadcast matmul — no view
/// ops — so fusion engines can match the matmul with its epilogue; turning the
/// broadcast batches into a single full matmul is the launch's job (see
/// `merged_rows` in burn-cubecl-fusion / burn-cubecl).
pub(crate) fn linear<B: Backend>(
    x: FloatTensor<B>,
    weight: FloatTensor<B>,
    bias: Option<FloatTensor<B>>,
) -> FloatTensor<B> {
    let ndims = x.shape().num_dims();

    // Reshape weight [d_input, d_output] -> [1, ..., 1, d_input, d_output] for batch matmul.
    let weight = unsqueeze_leading::<B>(weight, ndims);
    let output = B::float_matmul(x, weight);

    match bias {
        Some(bias) => {
            // Reshape bias [d_output] -> [1, ..., 1, d_output] to match output rank.
            let bias = unsqueeze_leading::<B>(bias, ndims);
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
/// transposed weight, so it shares the batched vec-mat transformation.
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
