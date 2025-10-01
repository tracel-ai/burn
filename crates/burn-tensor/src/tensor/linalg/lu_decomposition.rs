use crate::{Int, backend::Backend, cast::ToElement, linalg::swap_slices, s, tensor::Tensor};

/// The things that can go wrong when computing the LU decomposition.
#[derive(Debug)]
pub enum LuError {
    /// The input matrix is not square.
    NotSquareMatrix(String),
    /// The input matrix is singular.
    SingularMatrix(String),
}

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
pub fn lu_decomposition<B: Backend>(
    tensor: Tensor<B, 2>,
) -> Result<(Tensor<B, 2>, Tensor<B, 1, Int>), LuError> {
    let dims = tensor.shape().dims::<2>();
    if dims[0] != dims[1] {
        return Err(LuError::NotSquareMatrix(format!(
            "LU decomposition requires a square matrix, but got shape: {:?}",
            tensor.shape()
        )));
    }
    let n = dims[0];

    let mut permutations = Tensor::arange(0..n as i64, &tensor.device());
    let mut tensor = tensor;

    for k in 0..n {
        let mut p = k;
        let mut max = tensor.clone().slice(s![k, k]).abs();
        for i in (k + 1)..n {
            let val = tensor.clone().slice(s![i, k]).abs();
            if val.clone().greater(max.clone()).into_scalar().to_bool() {
                max = val.clone();
                p = i;
            }
        }

        // Avoid division by zero
        if max.into_scalar().to_f32().abs() < f32::EPSILON * 10.0 {
            return Err(LuError::SingularMatrix(format!(
                "LU decomposition failed: matrix is singular or near-singular at column {k}",
            )));
        }

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

    Ok((tensor, permutations))
}
