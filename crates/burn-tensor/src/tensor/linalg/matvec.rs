use crate::Numeric;
use crate::backend::Backend;
use crate::tensor::{BasicOps, Shape, Tensor};

/// Performs matrix-vector multiplication with optional batch dimensions.
///
/// The `matrix` tensor is expected to have rank `DM` with the last two dimensions representing
/// the matrix rows and columns. The `vector` tensor should have rank `DV = DM - 1`, sharing
/// broadcast-compatible batch dimensions and matching the last dimension of the matrix.
///
/// # Panics
///
/// * If the matrix rank is lower than 2.
/// * If the vector rank isn't one less than the matrix rank.
/// * If batch dimensions differ between the operands.
/// * If the inner dimensions are incompatible for multiplication.
pub fn matvec<B: Backend, const DM: usize, const DV: usize, K>(
    matrix: Tensor<B, DM, K>,
    vector: Tensor<B, DV, K>,
) -> Tensor<B, DV, K>
where
    K: BasicOps<B> + Numeric<B>,
{
    assert!(
        DM >= 2,
        "matvec expects the matrix to be at least rank 2 (got {DM})"
    );
    assert!(
        DM == DV + 1,
        "matvec expects the vector rank ({DV}) to be exactly one less than the matrix rank ({DM})",
    );

    let matrix_dims = matrix.shape().dims::<DM>();
    let vector_dims = vector.shape().dims::<DV>();

    // Validate batch dimensions (all leading dimensions prior to the matrix axes).
    let batch_rank = DM.saturating_sub(2);
    if batch_rank > 0 {
        let matrix_batch = Shape::from(&matrix_dims[..batch_rank]);
        let vector_batch = Shape::from(&vector_dims[..batch_rank]);

        assert!(
            matrix_batch.broadcast(&vector_batch).is_ok(),
            "Batch dimensions are not broadcast-compatible: matrix {:?} vs vector {:?}",
            &matrix_dims[..batch_rank],
            &vector_dims[..batch_rank]
        );
    }

    let matrix_inner = matrix_dims[DM - 1];
    let vector_inner = vector_dims[DV - 1];
    assert!(
        matrix_inner == vector_inner,
        "Inner dimension mismatch: matrix has {matrix_inner} columns but vector has {vector_inner} entries",
    );

    let vector_expanded = vector.unsqueeze_dim::<DM>(DV);
    matrix.matmul(vector_expanded).squeeze_dim::<DV>(DM - 1)
}
