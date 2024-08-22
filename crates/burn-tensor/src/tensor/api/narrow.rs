use crate::{
    backend::Backend, BasicOps, Bool, Dense, ReprPrimitive, TensorKind, TensorRepr, TensorStorage,
};
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
pub fn narrow<
    B: Backend,
    const D: usize,
    K: TensorKind<B> + BasicOps<B, SR>,
    SR: TensorStorage<B>,
>(
    tensor: ReprPrimitive<B, K, SR, D>,
    dim: usize,
    start: usize,
    length: usize,
) -> ReprPrimitive<B, K, SR, D>
where
    (B, K, SR): TensorRepr,
    (B, Bool, SR): TensorRepr,
{
    let shape = K::shape(&tensor);

    let ranges: Vec<_> = (0..D)
        .map(|i| {
            if i == dim {
                start..(start + length)
            } else {
                0..shape.dims[i]
            }
        })
        .collect();

    let ranges_array: [_; D] = ranges.try_into().unwrap();

    K::slice(tensor, ranges_array)
}
