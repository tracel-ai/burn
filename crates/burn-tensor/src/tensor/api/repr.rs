use crate::{backend::Backend, ops::SparseTensorOps};
use core::marker::PhantomData;

pub trait TensorRepr<B: Backend>: Clone + core::fmt::Debug {}

pub trait SparseRepr<B: Backend>: Clone + core::fmt::Debug + SparseTensorOps<Self, B> {
    type FloatPrimitive<const D: usize>: Clone + core::fmt::Debug + Send;
    type IntPrimitive<const D: usize>: Clone + core::fmt::Debug + Send;
    type BoolPrimitive<const D: usize>: Clone + core::fmt::Debug + Send;
}

#[derive(Clone, Debug)]
pub struct Dense;

#[derive(Clone, Debug)]
pub struct Sparse<R: SparseRepr<B>, B: Backend>(PhantomData<(R, B)>);

impl<B: Backend> TensorRepr<B> for Dense {}

impl<B: Backend, R: SparseRepr<B>> TensorRepr<B> for Sparse<R, B> {}
