use crate::backend::Backend;
use crate::tensor::{BasicOps, Tensor};
use crate::{Numeric, Shape};

/// Computes the outer product (and batched outer product) for rank-1 or rank-2 tensor.
///
/// Supported ranks:
/// - D = 1, R = 2: vectors (m,) × (n,) → (m, n)
/// - D = 2, R = 3: batched (b, m) × (b, n) → (b, m, n)
///
/// Panics:
/// - if D > 2
/// - if (D, R) is not (1,2) or (2,3)
/// - if D = 2 and batch dimensions differ
//
// Notes:
// - For large batched inputs, `x_col.matmul(y_row)` *might* be more performant
//   than broadcasted elemwise multiply; benchmarking needed to confirm.
pub fn outer<B: Backend, const D: usize, const R: usize, K>(
    x: Tensor<B, D, K>,
    y: Tensor<B, D, K>,
) -> Tensor<B, R, K>
where
    K: BasicOps<B> + Numeric<B>,
{
    if D == 1 {
        assert!(R == 2, "`outer` with D=1 must use R=2 (got R={})", R);
        let [m] = x.shape().dims();
        let [n] = y.shape().dims();

        let x_col = x.reshape(Shape::new([m, 1])); // (m, 1)
        let y_row = y.reshape(Shape::new([1, n])); // (1, n)

        x_col * y_row // (m, n)
    } else if D == 2 {
        assert!(R == 3, "`outer` with D=2 must use R=3 (got R={})", R);
        let [bx, m] = x.shape().dims();
        let [by, n] = y.shape().dims();
        assert_eq!(bx, by, "batch dimensions must match (got {} vs {})", bx, by);

        let x_col = x.reshape(Shape::new([bx, m, 1])); // (b, m, 1)
        let y_row = y.reshape(Shape::new([by, 1, n])); // (b, 1, n)

        x_col * y_row // (b, m, n)
    } else {
        panic!("`outer` only supports rank 1 or 2 tensors (got D={})", D);
    }
}
