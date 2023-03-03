use crate::backend::Backend;

pub struct Float;
pub struct Int;
pub struct Bool;

pub trait TensorKind<B: Backend> {
    type Primitive<const D: usize>;
}

impl<B: Backend> TensorKind<B> for Float {
    type Primitive<const D: usize> = B::TensorPrimitive<D>;
}

impl<B: Backend> TensorKind<B> for Int {
    type Primitive<const D: usize> = <B::IntegerBackend as Backend>::TensorPrimitive<D>;
}

impl<B: Backend> TensorKind<B> for Bool {
    type Primitive<const D: usize> = B::BoolTensorPrimitive<D>;
}
