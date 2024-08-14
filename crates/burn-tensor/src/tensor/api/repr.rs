use crate::{backend::Backend, ops::SparseTensorOps, TensorKind};
use core::marker::PhantomData;

pub trait TensorRepr<B: Backend>: Clone + core::fmt::Debug {
    fn name() -> &'static str;
}

pub trait SparseRepr<B: Backend>: Clone + core::fmt::Debug + SparseTensorOps<Self, B> {
    type FloatTensorPrimitive<const D: usize>: Clone + core::fmt::Debug + Send;
    type IntTensorPrimitive<const D: usize>: Clone + core::fmt::Debug + Send;
    type BoolTensorPrimitive<const D: usize>: Clone + core::fmt::Debug + Send;
    fn name() -> &'static str;
}

#[derive(Clone, Debug)]
pub struct Dense;

#[derive(Clone, Debug)]
pub struct Sparse<R: SparseRepr<B>, B: Backend>(PhantomData<(R, B)>);

impl<B: Backend> TensorRepr<B> for Dense {
    fn name() -> &'static str {
        "Dense"
    }
}

impl<R: SparseRepr<B>, B: Backend> TensorRepr<B> for Sparse<R, B> {
    fn name() -> &'static str {
        R::name()
    }
}
