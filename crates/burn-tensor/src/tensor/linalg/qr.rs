use crate::{check, check::TensorCheck, Tensor};
use alloc::vec;
use alloc::vec::Vec;

/// QR decomposition via Modified Gram-Schmidt with reorthogonalization.
///
/// Decomposes a matrix `A` (m x n, m >= n) into an orthonormal matrix `Q` (m x n)
/// and an upper triangular matrix `R` (n x n) such that `A ~= Q @ R`.
///
/// Uses Modified Gram-Schmidt with one Daniel-Gragg-Kaufman-Stewart (1976)
/// reorthogonalization pass for improved numerical stability. The operation
/// is composed from tensor primitives and works on any backend.
///
/// # Arguments
/// - `tensor` - An (m x n) matrix as a 2D tensor. Must satisfy `m >= n`.
///
/// # Returns
/// A tuple `(Q, R)`:
/// - `Q` - (m x n) tensor with orthonormal columns (`Q^T Q ~= I`).
/// - `R` - (n x n) upper triangular tensor.
///
/// # Panics
/// Panics if `m < n`.
///
/// # Performance note
/// This function extracts per-column norms from device memory. On backends
/// without fused scalar extraction this may cause device synchronizations.
/// A backend-kernel QR is tracked for future optimization.
///
/// # Example
/// ```rust,ignore
/// use burn_tensor::linalg::qr_decomposition;
///
/// fn example() {
///     let device = Default::default();
///     let a = Tensor::<2>::from_floats(
///         [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &device,
///     ).reshape([3, 2]);
///     let (q, r) = qr_decomposition(a);
///     // Q^T Q ~= I_2, A ~= Q @ R
/// }
/// ```
pub fn qr_decomposition(tensor: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
    let [m, n] = tensor.dims();
    check!(TensorCheck::qr_valid("qr_decomposition", m, n));

    let device = tensor.device();
    let mut q_cols: Vec<Tensor<2>> = Vec::with_capacity(n);
    let mut r_data = vec![0.0f32; n * n];

    for i in 0..n {
        let mut col = tensor.clone().slice([0..m, i..i + 1]);

        // Modified Gram-Schmidt: project onto previous q columns.
        for (j, q_j) in q_cols.iter().enumerate() {
            let r_ij = q_j
                .clone()
                .transpose()
                .matmul(col.clone())
                .into_scalar::<f32>();
            r_data[j * n + i] = r_ij;

            col = col.clone() - q_j.clone().mul_scalar(r_ij);
        }

        // Reorthogonalization pass.
        for q_j in q_cols.iter() {
            let coeff = q_j
                .clone()
                .transpose()
                .matmul(col.clone())
                .into_scalar::<f32>();

            col = col.clone() - q_j.clone().mul_scalar(coeff);
        }

        // Normalize.
        let norm = col
            .clone()
            .powf_scalar(2.0)
            .sum()
            .into_scalar::<f32>()
            .sqrt();

        r_data[i * n + i] = norm;

        col = col.div_scalar(norm);
        q_cols.push(col);
    }

    let r = Tensor::<1>::from_floats(r_data.as_slice(), &device).reshape([n, n]);
    (Tensor::cat(q_cols, 1), r)
}
