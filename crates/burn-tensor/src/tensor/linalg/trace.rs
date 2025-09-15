use super::diag;
use crate::backend::Backend;
use crate::tensor::Tensor;

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
/// Tensor with rank D - 1, where the last two matrix dimensions are replaced by a single
/// dimension containing the the trace for each matrix.
///
pub fn trace<B: Backend, const D: usize, const DO: usize>(tensor: Tensor<B, D>) -> Tensor<B, DO> {
    let diag_tensor = diag::<_, D, DO, _>(tensor);

    diag_tensor.sum_dim(DO - 1)
}
