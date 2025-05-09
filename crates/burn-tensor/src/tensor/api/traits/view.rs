use core::ops::Range;

use crate::{Shape, backend::Backend};

use super::TensorKind;

/// Trait that list all operations that operate on a tensor view.
/// A view shares the same underlying tensor data.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
///
/// # Note
pub trait ViewOps<B: Backend>: TensorKind<B> {
    // NOTE: reshape can either return a view or copy if conditions are compatible.
    // Notably, cubecl only returns a view if the tensor is already contiguous.
    // We should probably improve that to further specialize with shape compatibility.

    // TODO: movedim, split, chunk, squeeze, unsqueeze, narrow

    /// Transposes a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to transpose.
    ///
    /// # Returns
    ///
    /// The transposed tensor.
    fn transpose(tensor: Self::Primitive) -> Self::Primitive;

    /// Swaps two dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to swap the dimensions of.
    /// * `dim1` - The first dimension to swap.
    /// * `dim2` - The second dimension to swap.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions swapped.
    fn swap_dims(tensor: Self::Primitive, dim1: usize, dim2: usize) -> Self::Primitive;

    /// Permutes the dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to permute the dimensions of.
    /// * `axes` - The new order of the dimensions.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions permuted.
    fn permute(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive;

    ///  Select tensor elements corresponding for the given ranges.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `ranges` - The ranges of the elements to select.
    ///
    /// # Returns
    ///
    /// The selected elements.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For selecting elements of a tensor, users should prefer the [Tensor::slice](Tensor::slice) function,
    /// which is more high-level and designed for public use.
    fn slice(tensor: Self::Primitive, ranges: &[Range<usize>]) -> Self::Primitive;

    /// Broadcasts the given tensor to the specified shape.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to broadcast.
    /// * `shape` - The shape to broadcast to.
    ///
    /// # Returns
    ///
    /// The broadcasted tensor.
    fn expand(tensor: Self::Primitive, shape: Shape) -> Self::Primitive;
}
