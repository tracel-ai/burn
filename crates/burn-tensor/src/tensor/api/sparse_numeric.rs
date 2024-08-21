use crate::{backend::Backend, BasicOps, SparseRepr};

/// Trait that list all operations that can be applied on all numerical sparse tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait SparseNumeric<R: SparseRepr<B>, B: Backend>: BasicOps<B> {}
