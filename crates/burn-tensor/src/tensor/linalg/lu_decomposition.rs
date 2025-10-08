use crate::{
    Int, backend::Backend, cast::ToElement, check, check::TensorCheck, linalg::swap_slices, s,
    tensor::Tensor,
};
/// Performs PLU decomposition of a square matrix.
///
/// The function decomposes a given square matrix `A` into three matrices: a permutation vector `p`,
/// a lower triangular matrix `L`, and an upper triangular matrix `U`, such that `PA = LU`.
/// The permutation vector `p` represents the row swaps made during the decomposition process.
/// The lower triangular matrix `L` has ones on its diagonal and contains the multipliers used
/// during the elimination process below the diagonal. The upper triangular matrix `U` contains
/// the resulting upper triangular form of the matrix after the elimination process.
///
/// # Arguments
/// * `tensor` - A square matrix to decompose, represented as a 2D tensor.
///
/// # Returns
/// A tuple containing:
/// - A 2D tensor representing the combined `L` and `U` matrices.
/// - A 1D tensor representing the permutation vector `p`.
///
/// # Panics and numerical issues
/// - The function will panic if the input matrix is singular or near-singular.
/// - The function will panic if the input matrix is not square.
/// # Performance note (synchronization / device transfers)
/// This function may involve multiple synchronizations and device transfers, especially
/// when determining pivot elements and performing row swaps. This can impact performance,
pub fn lu_decomposition<B: Backend>(tensor: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1, Int>) {
    check!(TensorCheck::is_square::<2>(
        "lu_decomposition",
        &tensor.shape()
    ));
    let dims = tensor.shape().dims::<2>();
    let n = dims[0];

    let mut permutations = Tensor::arange(0..n as i64, &tensor.device());
    let mut tensor = tensor;

    for k in 0..n {
        // Find the pivot row
        let p = tensor
            .clone()
            .slice(s![k.., k])
            .abs()
            .argmax(0)
            .into_scalar()
            .to_usize()
            + k;
        let max = tensor.clone().slice(s![p, k]).abs();

        // Avoid division by zero
        let pivot = max.into_scalar();
        check!(TensorCheck::lu_decomposition_pivot::<B>(pivot));

        if p != k {
            tensor = swap_slices(tensor, s![k, ..], s![p, ..]);
            permutations = swap_slices(permutations, s![k], s![p]);
        }

        // Normalize k-th column under the diagonal
        if k < n - 1 {
            let a_kk = tensor.clone().slice(s![k, k]);
            let column = tensor.clone().slice(s![(k + 1).., k]) / a_kk;
            tensor = tensor.slice_assign(s![(k + 1).., k], column);
        }

        // Update the trailing submatrix
        for i in (k + 1)..n {
            // a[i, k+1..] -=  a[i, k] * a[k, k+1..]
            let a_ik = tensor.clone().slice(s![i, k]);
            let row_k = tensor.clone().slice(s![k, (k + 1)..]);
            let update = a_ik * row_k;
            let row_i = tensor.clone().slice(s![i, (k + 1)..]);
            tensor = tensor.slice_assign(s![i, (k + 1)..], row_i - update);
        }
    }

    (tensor, permutations)
}
