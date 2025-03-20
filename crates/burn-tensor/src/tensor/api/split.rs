use super::{TensorMetadata, narrow::narrow};
use crate::{BasicOps, TensorKind, backend::Backend};
use alloc::vec::Vec;

/// Splits the tensor along the given dimension into equally sized chunks (if possible)
/// with size `split_size`. Last chunk will be smaller if the tensor size along the given
/// dimension `dim` is not divisible by `split_size`.
///
/// # Arguments
///
/// * `tensor` - The tensor.
/// * `split_size` - The size of a single chunk.
/// * `dim` - The dimension along which to split the tensor.
///
/// # Returns
///
/// A vector of tensors.
///
/// # Remarks
///
/// This (and the following) are fallback solutions that is used only when the backend doesn't have the corresponding implementation.
/// Ideally, it is supposed to be implemented by the backend and the backend implementation will be resolved
/// by static dispatch. It is not designed for direct usage by users, and not recommended to import
/// or use this function directly.
pub fn split<B: Backend, K: TensorKind<B> + BasicOps<B>>(
    tensor: K::Primitive,
    split_size: usize,
    dim: usize,
) -> Vec<K::Primitive> {
    let size = tensor.shape().dims[dim];
    let mut tensors = Vec::new();

    let mut start = 0;
    while start < size {
        let length = usize::min(split_size, size - start);
        tensors.push(narrow::<B, K>(tensor.clone(), dim, start, length));
        start += length;
    }

    tensors
}

/// Splits the tensor along the given dimension into chunks with sizes in
/// `dim` according to `split_sizes`.
///
/// # Arguments
///
/// * `tensor` - The tensor.
/// * `split_sizes` - Vector of sizes for each chunk.
/// * `dim` - The dimension along which to split the tensor.
///
/// # Returns
///
/// A vector of tensors.
///
/// # Remarks
///
/// Fallback solution for backends with no equivalent functionality.
pub fn split_with_sizes<B: Backend, K: TensorKind<B> + BasicOps<B>>(
    tensor: K::Primitive,
    split_sizes: Vec<usize>,
    dim: usize,
) -> Vec<K::Primitive> {
    let mut tensors = Vec::new();

    let mut start = 0;
    for length in split_sizes {
        if length == 0 {
            continue;
        }
        tensors.push(narrow::<B, K>(tensor.clone(), dim, start, length));
        start += length;
    }

    tensors
}
