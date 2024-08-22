use crate::{backend::Backend, ops::SparseTensorOps, Bool, Float, Int, Tensor, TensorKind};
use core::marker::PhantomData;

pub trait TensorStorage<B: Backend>: Clone + core::fmt::Debug {
    fn name() -> &'static str;
}

pub trait SparseStorage<B: Backend>: Clone + core::fmt::Debug + SparseTensorOps<Self, B> {
    type SparsePrimitive<K: TensorKind<B>, const D: usize>: Clone + core::fmt::Debug + Send;

    fn name() -> &'static str;
}

#[derive(Clone, Debug)]
pub struct Dense;

#[derive(Clone, Debug)]
pub struct Sparse<B: Backend, SR: SparseStorage<B>>(PhantomData<(B, SR)>);

impl<B: Backend> TensorStorage<B> for Dense {
    fn name() -> &'static str {
        "Dense"
    }
}

impl<B: Backend, SR: SparseStorage<B>> TensorStorage<B> for Sparse<B, SR> {
    fn name() -> &'static str {
        SR::name()
    }
}
