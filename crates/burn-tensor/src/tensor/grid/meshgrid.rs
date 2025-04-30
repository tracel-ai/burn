use crate::backend::Backend;
use crate::tensor::grid::{GridIndexing, GridOptions, GridSparsity};
use crate::tensor::{BasicOps, Tensor};
use alloc::vec::Vec;

/// Return a collection of coordinate matrices for coordinate vectors.
///
/// Takes N 1D tensors and returns N tensors where each tensor represents the coordinates
/// in one dimension across an N-dimensional grid.
///
/// Based upon `options.sparse`, the generated coordinate tensors can either be `Sparse` or `Dense`:
/// * In `Sparse` mode, output tensors will have shape 1 everywhere except their cardinal dimension.
/// * In `Dense` mode, output tensors will be expanded to the full grid shape.
///
/// Based upon `options.indexing`, the generated coordinate tensors will use either:
/// * `Matrix` indexing, where dimensions are in the same order as their cardinality.
/// * `Cartesian` indexing; where the first two dimensions are swapped.
///
/// See:
///  - https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
///  - https://pytorch.org/docs/stable/generated/torch.meshgrid.html
///
/// # Arguments
///
/// * `tensors` - A slice of 1D tensors
/// * `options` - the options.
///
/// # Returns
///
/// A vector of N N-dimensional tensors representing the grid coordinates.
pub fn meshgrid<B: Backend, const N: usize, K, O>(
    tensors: &[Tensor<B, 1, K>; N],
    options: O,
) -> [Tensor<B, N, K>; N]
where
    K: BasicOps<B>,
    O: Into<GridOptions>,
{
    let options = options.into();
    let swap_dims = options.indexing == GridIndexing::Cartesian && N > 1;

    let grid_shape: [usize; N] = tensors
        .iter()
        .map(|t| t.dims()[0])
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let result = tensors
        .iter()
        .enumerate()
        .map(|(i, tensor)| {
            let mut coord_tensor_shape = [1; N];
            coord_tensor_shape[i] = grid_shape[i];

            // Reshape the tensor to have singleton dimensions in all but the i-th dimension
            let mut tensor = tensor.clone().reshape(coord_tensor_shape);

            if options.sparsity == GridSparsity::Dense {
                tensor = tensor.expand(grid_shape);
            }

            if swap_dims {
                // Swap the first two dimensions for "xy" / CartesianIndexing
                tensor = tensor.swap_dims(0, 1);
            }

            tensor
        })
        .collect::<Vec<_>>();

    result.try_into().unwrap()
}
