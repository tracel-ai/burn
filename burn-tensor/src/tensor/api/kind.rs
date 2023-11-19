use crate::backend::Backend;

/// A type-level representation of the kind of a float tensor
#[derive(Clone, Debug)]
pub struct Float;

/// A type-level representation of the kind of a int tensor.
#[derive(Clone, Debug)]
pub struct Int;

/// A type-level representation of the kind of a bool tensor.
#[derive(Clone, Debug)]
pub struct Bool;

/// A type-level representation of the kind of a tensor.
pub trait TensorKind<B: Backend>: Clone + core::fmt::Debug {
    /// The primitive type of the tensor.
    type Primitive<const D: usize>: Clone + core::fmt::Debug;

    /// The name of the tensor kind.
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
