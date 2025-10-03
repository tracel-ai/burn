mod cosine_similarity;
mod diag;
mod lu_decomposition;
mod outer;
mod trace;
mod vector_norm;

pub use cosine_similarity::*;
pub use diag::*;
pub use lu_decomposition::*;
pub use outer::*;
pub use trace::*;
pub use vector_norm::*;

use crate::{BasicOps, SliceArg, Tensor, TensorKind, backend::Backend};

/// Swaps two slices of a tensor.
/// # Arguments
/// * `tensor` - The input tensor.
/// * `slices1` - The first slice to swap.
/// * `slices2` - The second slice to swap.
/// # Returns
/// A new tensor with the specified slices swapped.
/// # Notes
/// This method will be usueful for matrix factorization algorithms.
fn swap_slices<B: Backend, const D: usize, K, S>(
    tensor: Tensor<B, D, K>,
    slices1: S,
    slices2: S,
) -> Tensor<B, D, K>
where
    S: SliceArg<D> + Clone,
    K: TensorKind<B> + BasicOps<B>,
{
    let temporary = tensor.clone().slice(slices1.clone());
    let tensor = tensor
        .clone()
        .slice_assign(slices1, tensor.slice(slices2.clone()));
    tensor.slice_assign(slices2, temporary)
}
