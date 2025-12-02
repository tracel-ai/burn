use crate::{
    AutodiffBackend,
    tensor::{BasicOps, TensorKind},
};

/// Trait that list all operations that can be applied on all tensors on an autodiff backend.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait BasicAutodiffOps<B: AutodiffBackend>: BasicOps<B> + BasicOps<B::InnerBackend> {
    /// Inner primitive tensor.
    type InnerKind: BasicOps<B::InnerBackend>;

    /// Returns the inner tensor without the autodiff information.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the [Tensor::inner](Tensor::inner) function,
    /// which is more high-level and designed for public use.
    fn inner(
        tensor: <Self as TensorKind<B>>::Primitive,
    ) -> <Self::InnerKind as TensorKind<B::InnerBackend>>::Primitive;

    /// Convert a tensor to the autodiff backend.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the [Tensor::from_inner](Tensor::from_inner) function,
    /// which is more high-level and designed for public use.
    fn from_inner(
        inner: <Self::InnerKind as TensorKind<B::InnerBackend>>::Primitive,
    ) -> <Self as TensorKind<B>>::Primitive;
}
