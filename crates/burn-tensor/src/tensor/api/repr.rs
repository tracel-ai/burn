use crate::{
    backend::Backend, Dense, Float, Sparse, SparseStorage, Tensor, TensorKind, TensorStorage,
};

pub type ReprPrimitive<B, K, S, const D: usize> = <(B, K, S) as TensorRepr>::Primitive<D>;

pub trait TensorRepr {
    type Primitive<const D: usize>: Clone + core::fmt::Debug + Send;
}

impl<B: Backend, K: TensorKind<B>> TensorRepr for (B, K, Dense) {
    type Primitive<const D: usize> = K::Primitive<D>;
}

impl<B: Backend, K: TensorKind<B>, SR: SparseStorage<B>> TensorRepr for (B, K, Sparse<B, SR>) {
    type Primitive<const D: usize> = SR::SparsePrimitive<K, D>;
}
