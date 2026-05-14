use burn_backend::{TensorMetadata, TensorPrimitive};
use burn_dispatch::{Dispatch, DispatchTensor};

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
pub trait TensorKind: Clone + core::fmt::Debug {
    /// The primitive type of the tensor.
    type Primitive: TensorMetadata;

    /// The name of the tensor kind.
    fn name() -> &'static str;
}

impl TensorKind for Float {
    type Primitive = FloatTensor;
    fn name() -> &'static str {
        "Float"
    }
}

impl TensorKind for Int {
    type Primitive = IntTensor;
    fn name() -> &'static str {
        "Int"
    }
}

impl TensorKind for Bool {
    type Primitive = BoolTensor;
    fn name() -> &'static str {
        "Bool"
    }
}

// Tensor primitive type aliases
pub(crate) type FloatTensor = TensorPrimitive<Dispatch>;
pub(crate) type IntTensor = DispatchTensor;
pub(crate) type BoolTensor = DispatchTensor;
