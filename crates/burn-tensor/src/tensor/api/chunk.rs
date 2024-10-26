use super::narrow::narrow;
use crate::{backend::Backend, BasicOps, TensorKind};
use alloc::vec::Vec;

/// Split the tensor along the given dimension into chunks.
///
/// # Arguments
///
/// * `tensor` - The tensor.
/// * `chunks` - The number of chunks to be produced
/// * `times` - The dimension along which the tensor will be split.
///
/// # Returns
///
/// A vectors of tensors
///
/// # Remarks
///
/// This is a fallback solution that used only when the backend doesn't have the corresponding implementation.
/// Ideally, it is supposed to be implemented by the backend and the backend implementation will be resolved
/// by static dispatch. It is not designed for direct usage by users, and not recommended to import
/// or use this function directly.
pub fn chunk<B: Backend, K: TensorKind<B> + BasicOps<B>>(
    tensor: K::Primitive,
    chunks: usize,
    dim: usize,
) -> Vec<K::Primitive> {
    let size = K::shape(&tensor).dims[dim];
    if size < chunks {
        return (0..size)
            .map(|i| narrow::<B, K>(tensor.clone(), dim, i, 1))
            .collect();
    }

    let mut tensors = Vec::with_capacity(chunks);
    let mut sum_chunk_size = 0;
    if size % chunks == 0 {
        let chunk_size = size / chunks;
        for _ in 0..chunks {
            tensors.push(narrow::<B, K>(
                tensor.clone(),
                dim,
                sum_chunk_size,
                chunk_size,
            ));
            sum_chunk_size += chunk_size;
        }
    } else {
        let chunk_size = (size / chunks) + 1; // assumes not divisible
        for _ in 0..chunks - 1 {
            tensors.push(narrow::<B, K>(
                tensor.clone(),
                dim,
                sum_chunk_size,
                chunk_size,
            ));
            sum_chunk_size += chunk_size;
        }
        let remainder = size % chunk_size;
        tensors.push(narrow::<B, K>(
            tensor.clone(),
            dim,
            sum_chunk_size,
            remainder,
        ));
    }

    tensors
}
