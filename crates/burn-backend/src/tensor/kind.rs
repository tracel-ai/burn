use crate::{Backend, TensorMetadata, TensorPrimitive};

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
/// Metadata access is lazy.
pub trait TensorKind<B: Backend>: Clone + core::fmt::Debug {
    /// The primitive type of the tensor.
    type Primitive: TensorMetadata;

    /// The name of the tensor kind.
    fn name() -> &'static str;
}

impl<B: Backend> TensorKind<B> for Float {
    type Primitive = TensorPrimitive<B>;
    fn name() -> &'static str {
        "Float"
    }
}

impl<B: Backend> TensorKind<B> for Int {
    type Primitive = B::IntTensorPrimitive;
    fn name() -> &'static str {
        "Int"
    }
}

impl<B: Backend> TensorKind<B> for Bool {
    type Primitive = B::BoolTensorPrimitive;
    fn name() -> &'static str {
        "Bool"
    }
}
