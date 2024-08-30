use crate::check;

use crate::{
    backend::Backend, check::TensorCheck, BasicOps, Bool, Element, ElementConversion, Int, Shape,
    Sparse, SparseStorage, Tensor, TensorKind, TensorRepr,
};

/// Trait that list all operations that can be applied on all sparse numerical tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait SparseNumeric<B, K, SR>: TensorRepr
where
    B: Backend,
    K: TensorKind<B> + BasicOps<B>,
    SR: SparseStorage<B>,
    K::Elem: Element,
{
}

impl<B, const D: usize, K, SR> Tensor<B, D, K, Sparse<B, SR>>
where
    B: Backend,
    K: TensorKind<B> + BasicOps<B>,
    SR: SparseStorage<B>,
    (B, K, SR): SparseNumeric<B, K, SR>,
    K::Elem: Element,
{
}
