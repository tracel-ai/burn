mod cosine_similarity;
mod diag;
mod lu_decomposition;
mod matvec;
mod outer;
mod trace;
mod vector_norm;

pub use cosine_similarity::*;
pub use diag::*;
pub use lu_decomposition::*;
pub use matvec::*;
pub use outer::*;
pub use trace::*;
pub use vector_norm::*;

use crate::{SliceArg, Tensor, kind::Basic};

/// Swaps two slices of a tensor.
/// # Arguments
/// * `tensor` - The input tensor.
/// * `slices1` - The first slice to swap.
/// * `slices2` - The second slice to swap.
/// # Returns
/// A new tensor with the specified slices swapped.
/// # Notes
/// This method will be useful for matrix factorization algorithms.
fn swap_slices<const D: usize, K, S>(tensor: Tensor<D, K>, slices1: S, slices2: S) -> Tensor<D, K>
where
    S: SliceArg + Clone,
    K: Basic,
{
    let temporary = tensor.clone().slice(slices1.clone());
    let tensor = tensor
        .clone()
        .slice_assign(slices1, tensor.slice(slices2.clone()));
    tensor.slice_assign(slices2, temporary)
}
