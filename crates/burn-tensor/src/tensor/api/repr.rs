use crate::{backend::Backend, Dense, Float, Sparse, SparseStorage, TensorKind, TensorStorage};

pub type ReprPrimitive<B, K, S, const D: usize> =
    <TensorRepr as TensorReprT<B, K, S>>::Primitive<D>;

pub trait TensorReprT<B: Backend, K: TensorKind<B>, S: TensorStorage<B>> {
    type Primitive<const D: usize>: Clone + core::fmt::Debug + Send;
}

pub struct TensorRepr;

impl<B: Backend, K: TensorKind<B>> TensorReprT<B, K, Dense> for TensorRepr {
    type Primitive<const D: usize> = K::Primitive<D>;
}

impl<B: Backend, K: TensorKind<B>, SR: SparseStorage<B>> TensorReprT<B, K, Sparse<B, SR>>
    for TensorRepr
{
    type Primitive<const D: usize> = SR::SparsePrimitive<K, D>;
}
