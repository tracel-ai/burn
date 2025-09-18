use crate::backend::Backend;
use crate::check;
use crate::check::TensorCheck;
use crate::tensor::{Int, Shape, Tensor};
use crate::{BasicOps, TensorKind};

/// Returns the diag of a matrix.
///
/// For batched inputs, returns of each matrix in the batch independently.
///
/// The diag operation extracts the diagonal elements of the last two dimensions,
/// treating them as the matrix dimensions, while preserving all leading batch dimensions.
///
/// # Arguments
///
/// * `tensor` - The input tensor with at least 2 dimensions.
///
/// # Returns
/// A tensor of rank `D - 1`, where the last dimension contains the diagonal elements of the input.
pub fn diag<B: Backend, const D: usize, const DO: usize, K>(
    tensor: Tensor<B, D, K>,
) -> Tensor<B, DO, K>
where
    K: TensorKind<B> + BasicOps<B>,
{
    check!(TensorCheck::diag::<D, DO>());

    let shape = tensor.shape();
    let rows = shape.dims[D - 2];
    let cols = shape.dims[D - 1];
    let diag_len = rows.min(cols);
    let device = tensor.device();

    // create the indices for the diag
    let mut flat_shape = shape.dims.clone();
    flat_shape[D - 2] = rows * cols;
    flat_shape[D - 1] = 1;
    let flat: Tensor<B, D, K> = tensor.reshape(Shape::from(flat_shape));

    let range = Tensor::<B, 1, Int>::arange(0..diag_len as i64, &device);
    let step_tensor = Tensor::<B, 1, Int>::from_data([cols as i64 + 1], &device);
    let indices = range * step_tensor;
    flat.take::<1, D>(D - 2, indices).squeeze(D - 1)
}
