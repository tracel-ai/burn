use crate::backend::Backend;
use crate::tensor::{BasicOps, Tensor};
use crate::{Numeric, Shape};

/// Computes the batch outer product of two 2D tensors (batched vectors).
///
/// Given `x` of shape `(batch, m)` and `y` of shape `(batch, n)`,
/// returns a tensor of shape `(batch, m, n)` where
/// `out[b, i, j] = x[b, i] * y[b, j]`.
///
/// # Arguments
/// * `x` - Batched vectors of shape `(batch, m)`.
/// * `y` - Batched vectors of shape `(batch, n)`.
///
/// # Returns
/// * A 3D tensor of shape `(batch, m, n)`.
pub fn outer_batch<B: Backend, K>(x: Tensor<B, 2, K>, y: Tensor<B, 2, K>) -> Tensor<B, 3, K>
where
    K: BasicOps<B> + Numeric<B>,
{
    let [batch_x, m] = x.shape().dims();
    let [batch_y, n] = y.shape().dims();
    assert_eq!(batch_x, batch_y, "batch dimensions must match");

    let x_col = x.reshape(Shape::new([batch_x, m, 1])); // (batch, m, 1)
    let y_row = y.reshape(Shape::new([batch_y, 1, n])); // (batch, 1, n)

    x_col * y_row // (batch, m, n)
}
