use crate::{backend::Backend, ops::SparseTensorOps, Bool, Float, Int, Tensor, TensorKind};
use core::marker::PhantomData;

pub trait TensorRepr<B: Backend>: Clone + core::fmt::Debug {
    type Primitive<K: TensorKind<B>, const D: usize>: Clone + core::fmt::Debug + Send;

    fn name() -> &'static str;
}

pub trait ChangeRepr<B: Backend, R: TensorRepr<B>>: TensorRepr<B> {
    fn change_repr<const D: usize, K: TensorKind<B, Self>, K2: TensorKind<B, R>>(
        lhs: Tensor<B, D, K, Self>,
    ) -> Tensor<B, D, K2, R>;
}

pub trait SparseRepr<B: Backend>: Clone + core::fmt::Debug + SparseTensorOps<Self, B> {
    type Primitive<K: TensorKind<B>, const D: usize>: Clone + core::fmt::Debug + Send;

    fn name() -> &'static str;
}

#[derive(Clone, Debug)]
pub struct Dense;

#[derive(Clone, Debug)]
pub struct Sparse<R: SparseRepr<B>, B: Backend>(PhantomData<(R, B)>);

impl<B: Backend> TensorRepr<B> for Dense {
    type Primitive<K: TensorKind<B>, const D: usize> = K::DensePrimitive<D>;

    fn name() -> &'static str {
        "Dense"
    }
}

impl<R: SparseRepr<B>, B: Backend> TensorRepr<B> for Sparse<R, B> {
    type Primitive<K: TensorKind<B>, const D: usize> = R::Primitive<K, D>;

    fn name() -> &'static str {
        R::name()
    }
}
