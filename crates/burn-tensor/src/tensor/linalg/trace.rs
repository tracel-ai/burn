use crate::backend::Backend;
use crate::check;
use crate::check::TensorCheck;
use crate::tensor::Shape;
use crate::tensor::{Bool, Tensor};

/// Computes the trace of the of square matrices.
///
/// For batched inputs, computes the trace of each matrix in the batch independently.
///
/// The trace operation sums the diagonal elements of the last two dimensions,
/// treating them as the matrix dimensions, while preserving all leading batch dimensions.
///
/// # Arguments
///
/// * `tensor` - The input tensor with at least 2 dimensions.
///
/// # Returns
///
/// Tensor with same shape as the input, where the last two dimensions are reduced to size 1,
/// containing the trace of each matrix.
///
pub fn trace<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    check!(TensorCheck::tri::<D>());

    let shape = tensor.shape();
    let mat_shape = [shape.dims[D - 2], shape.dims[D - 1]];

    let mask = Tensor::<B, 2, Bool>::diag_mask(mat_shape, 0, &tensor.device());

    // Handle broadcasting of the mask
    let mut mask_shape = shape.dims.clone();
    for item in mask_shape.iter_mut().take(D - 2) {
        *item = 1;
    }
    mask_shape[D - 2] = mat_shape[0];
    mask_shape[D - 1] = mat_shape[1];
    let mask_shape = Shape::from(mask_shape);
    let mask = mask.reshape(mask_shape);

    tensor.mask_fill(mask, 0).sum_dim(D - 1).sum_dim(D - 2)
}
