use crate::backend::Backend;
use crate::tensor::grid::{GridCompatIndexing, GridSparsity};
use crate::tensor::{BasicOps, Tensor};
use alloc::vec::Vec;

/// Return a collection of coordinate matrices for coordinate vectors.
///
/// Takes N 1D tensors and returns N tensors where each tensor represents the coordinates
/// in one dimension across an N-dimensional grid.
///
/// The generated coordinate tensors can either be `Sparse` or `Dense`:
/// * In `Sparse` mode, output tensors will have shape 1 everywhere except their cardinal dimension.
/// * In `Dense` mode, output tensors will be expanded to the full grid shape.
///
/// Equivalent to ``meshgrid_compat(tensors, sparsity, GridCompatIndexing::CartesianIndexing)``.
///
/// Users who need compatibility with the legacy `"xy"` indexing mode from
/// NumPy and PyTorch should use `meshgrid_compat`.
///
/// See:
///  - https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
///  - https://pytorch.org/docs/stable/generated/torch.meshgrid.html
///
/// # Arguments
///
/// * `tensors` - A slice of 1D tensors
/// * `sparsity` - the sparse mode.
/// * `indexing` - Optional IndexingMode, defaults to `MatrixIndexing`.
///
/// # Returns
///
/// A vector of N N-dimensional tensors representing the grid coordinates.
pub fn meshgrid<B: Backend, const N: usize, K>(
    tensors: &[Tensor<B, 1, K>; N],
    sparsity: GridSparsity,
) -> [Tensor<B, N, K>; N]
where
    K: BasicOps<B>,
{
    meshgrid_compat(tensors, sparsity, GridCompatIndexing::MatrixIndexing)
}

/// Return a dense collection of coordinate matrices for coordinate vectors.
///
/// Takes N 1D tensors and returns N dense tensors (populated with the full
/// cartesian product shape of the inputs) where each tensor represents
/// the coordinates in one dimension across an N-dimensional grid.
///
/// Equivalent to ``meshgrid(tensors, GridSparsity::Dense)``.
///
/// Users who need compatibility with the legacy `"xy"` indexing mode from
/// NumPy and PyTorch should use `meshgrid_compat`.
///
/// See:
///  - https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
///  - https://pytorch.org/docs/stable/generated/torch.meshgrid.html
///
/// # Arguments
///
/// * `tensors` - A slice of 1D tensors
/// * `sparsity` - the sparse mode.
/// * `indexing` - Optional IndexingMode, defaults to `MatrixIndexing`.
///
/// # Returns
///
/// A vector of N N-dimensional tensors representing the grid coordinates.
pub fn meshgrid_dense<B: Backend, const N: usize, K>(
    tensors: &[Tensor<B, 1, K>; N],
) -> [Tensor<B, N, K>; N]
where
    K: BasicOps<B>,
{
    meshgrid_compat(
        tensors,
        GridSparsity::Dense,
        GridCompatIndexing::MatrixIndexing,
    )
}

/// Return a collection of coordinate matrices for coordinate vectors.
///
/// Takes N 1D tensors and returns N sparse tensors (each tensor has shape 1
/// everywhere except its cardinal dimension) where each tensor represents
/// the coordinates in one dimension across an N-dimensional grid.
///
/// Equivalent to ``meshgrid(tensors, GridSparsity::Sparse)``.
///
/// Users who need compatibility with the legacy `"xy"` indexing mode from
/// NumPy and PyTorch should use `meshgrid_compat`.
///
/// See:
///  - https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
///  - https://pytorch.org/docs/stable/generated/torch.meshgrid.html
///
/// # Arguments
///
/// * `tensors` - A slice of 1D tensors
/// * `sparsity` - the sparse mode.
/// * `indexing` - Optional IndexingMode, defaults to `MatrixIndexing`.
///
/// # Returns
///
/// A vector of N N-dimensional tensors representing the grid coordinates.
pub fn meshgrid_sparse<B: Backend, const N: usize, K>(
    tensors: &[Tensor<B, 1, K>; N],
) -> [Tensor<B, N, K>; N]
where
    K: BasicOps<B>,
{
    meshgrid_compat(
        tensors,
        GridSparsity::Sparse,
        GridCompatIndexing::MatrixIndexing,
    )
}

/// Return a collection of coordinate matrices for coordinate vectors.
///
/// Takes N 1D tensors and returns N tensors where each tensor represents the coordinates
/// in one dimension across an N-dimensional grid.
///
/// The generated coordinate tensors can either be `Sparse` or `Dense`:
/// * In `Sparse` mode, output tensors will have shape 1 everywhere except their cardinal dimension.
/// * In `Dense` mode, output tensors will be expanded to the full grid shape.
///
/// The optional `indexing` argument allows you to choose between two indexing modes:
/// * `MatrixIndexing` (default): Dimensions are in the same order as the cardinality of the inputs.
/// * `CartesianIndexing`: The first two dimensions are swapped, giving a backward compatibility
///   with the legacy Cartesian indexing ("xy") style used by NumPy and PyTorch.
///
/// See:
///  - https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
///  - https://pytorch.org/docs/stable/generated/torch.meshgrid.html
///
/// # Arguments
///
/// * `tensors` - A slice of 1D tensors
/// * `sparsity` - the sparse mode.
/// * `indexing` - Optional IndexingMode, defaults to `MatrixIndexing`.
///
/// # Returns
///
/// A vector of N N-dimensional tensors representing the grid coordinates.
pub fn meshgrid_compat<B: Backend, const N: usize, K>(
    tensors: &[Tensor<B, 1, K>; N],
    sparsity: GridSparsity,
    indexing: impl Into<Option<GridCompatIndexing>>,
) -> [Tensor<B, N, K>; N]
where
    K: BasicOps<B>,
{
    let indexing_mode = indexing
        .into()
        .unwrap_or(GridCompatIndexing::MatrixIndexing);
    let swap_dims = indexing_mode == GridCompatIndexing::CartesianIndexing && N > 1;

    let mut grid_shape = [0; N];
    for (i, tensor) in tensors.iter().enumerate() {
        assert_eq!(
            tensor.dims().len(),
            1,
            "All tensors must be 1D, found shape: {:?}",
            tensor.dims()
        );
        grid_shape[i] = tensor.dims()[0];
    }

    let result = tensors
        .iter()
        .enumerate()
        .map(|(i, tensor)| {
            let mut coord_tensor_shape = [1; N];
            coord_tensor_shape[i] = grid_shape[i];

            // Reshape the tensor to have singleton dimensions in all but the i-th dimension
            let mut tensor = tensor.clone().reshape(coord_tensor_shape);

            if sparsity == GridSparsity::Dense {
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
