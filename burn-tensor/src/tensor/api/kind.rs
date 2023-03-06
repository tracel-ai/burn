use crate::backend::Backend;

#[derive(Clone, Debug)]
pub struct Float;
#[derive(Clone, Debug)]
pub struct Int;
#[derive(Clone, Debug)]
pub struct Bool;

pub trait TensorKind<B: Backend>: Clone {
    type Primitive<const D: usize>: Clone + core::fmt::Debug;
}

impl<B: Backend> TensorKind<B> for Float {
    type Primitive<const D: usize> = B::TensorPrimitive<D>;
}

impl<B: Backend> TensorKind<B> for Int {
    type Primitive<const D: usize> = B::IntTensorPrimitive<D>;
}

impl<B: Backend> TensorKind<B> for Bool {
    type Primitive<const D: usize> = B::BoolTensorPrimitive<D>;
}
