use crate::{bridge::BasicOps, ops::BridgeTensor};

/// Trait that list all operations that can be applied on all tensors on an autodiff backend.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by the [`Tensor`](crate::Tensor) struct.
pub(crate) trait BasicAutodiffOps: BasicOps {
    /// Returns the inner tensor without the autodiff information.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the [`Tensor::inner`](crate::Tensor::inner)
    /// function, which is more high-level and designed for public use.
    fn inner(tensor: BridgeTensor) -> BridgeTensor;

    /// Convert a tensor to the autodiff backend.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the [`Tensor::from_inner`](crate::Tensor::from_inner)
    /// function, which is more high-level and designed for public use.
    fn from_inner(inner: BridgeTensor) -> BridgeTensor;
}
