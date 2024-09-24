use crate::{backend::Backend, BasicOps, TensorKind};
use alloc::vec::Vec;

/// Returns a new tensor with the given dimension narrowed to the given range.
///
/// # Arguments
///
/// * `tensor` - The tensor.
/// * `dim` - The dimension along which the tensor will be narrowed.
/// * `start` - The starting point of the given range.
/// * `length` - The ending point of the given range.
/// # Panics
///
/// - If the dimension is greater than the number of dimensions of the tensor.
/// - If the given range exceeds the number of elements on the given dimension.
///
/// # Returns
///
/// A new tensor with the given dimension narrowed to the given range.
pub fn narrow<B: Backend, K: TensorKind<B> + BasicOps<B>>(
    tensor: K::Primitive,
    dim: usize,
    start: usize,
    length: usize,
) -> K::Primitive {
    let shape = K::shape(&tensor);

    let ranges: Vec<_> = shape
        .dims
        .iter()
        .enumerate()
        .map(|(i, d)| {
            if i == dim {
                start..(start + length)
            } else {
                0..*d
            }
        })
        .collect();

    K::slice(tensor, &ranges)
}
