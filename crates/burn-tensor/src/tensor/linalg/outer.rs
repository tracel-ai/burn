use crate::backend::Backend;
use crate::tensor::{BasicOps, Tensor};
use crate::{Numeric, Shape};
/// Computes the outer product of two 1D tensors (vectors).
///
/// Given `x` of shape `(m,)` and `y` of shape `(n,)`, returns a tensor
/// of shape `(m, n)` where `out[i, j] = x[i] * y[j]`.
///
/// See:
/// - [torch.outer](https://pytorch.org/docs/stable/generated/torch.outer.html)
/// - [numpy.outer](https://numpy.org/doc/stable/reference/generated/numpy.outer.html)
///
/// # Arguments
///
/// * `x` - A 1D input tensor.
/// * `y` - A 1D input tensor.
///
/// # Returns
///
/// A 2D tensor containing the outer product of `x` and `y`.
pub fn outer<B: Backend, K>(
    x: Tensor<B, 1, K>, // x is a 1-D vector (rank 1)
    y: Tensor<B, 1, K>, // y is a 1-D vector (rank 1)
) -> Tensor<B, 2, K>
// we return a 2-D matrix (rank 2)
where
    K: BasicOps<B> + Numeric<B>, // works for floats AND ints
{
    // x.shape().dims() returns an array of dimension sizes.
    // because x is rank-1, it's exactly [m].
    let [m] = x.shape().dims();
    let [n] = y.shape().dims();

    // reshape x to (m, 1) => column
    let x_col = x.reshape(Shape::new([m, 1]));

    // reshape y to (1, n) => row
    let y_row = y.reshape(Shape::new([1, n]));

    // broadcasted multiply:
    // (m,1) * (1,n) -> (m,n), with M[i,j] = x[i] * y[j]
    x_col * y_row
}
