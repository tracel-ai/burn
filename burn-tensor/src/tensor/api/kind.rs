use crate::backend::Backend;

#[derive(Clone, Debug)]
pub struct Float;
#[derive(Clone, Debug)]
pub struct Int;
#[derive(Clone, Debug)]
pub struct Bool;

pub trait TensorKind<B: Backend>: Clone + core::fmt::Debug {
    type Primitive<const D: usize>: Clone + core::fmt::Debug;
    fn name() -> &'static str;
}

impl<B: Backend> TensorKind<B> for Float {
    type Primitive<const D: usize> = B::TensorPrimitive<D>;
    fn name() -> &'static str {
        "Float"
    }
}

impl<B: Backend> TensorKind<B> for Int {
    type Primitive<const D: usize> = B::IntTensorPrimitive<D>;
    fn name() -> &'static str {
        "Int"
    }
}

impl<B: Backend> TensorKind<B> for Bool {
    type Primitive<const D: usize> = B::BoolTensorPrimitive<D>;
    fn name() -> &'static str {
        "Bool"
    }
}
