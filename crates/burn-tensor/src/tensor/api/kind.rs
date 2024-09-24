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

#[derive(Debug, Clone)]
/// A primitive tensor representation.
pub enum TensorPrimitive<B: Backend> {
    /// Float tensor primitive.
    Float(B::FloatTensorPrimitive),
    /// Quantized float tensor primitive.
    QFloat(B::QuantizedTensorPrimitive),
}

impl<B: Backend> TensorPrimitive<B> {
    /// Returns the full tensor representation.
    pub fn tensor(self) -> B::FloatTensorPrimitive {
        match self {
            Self::QFloat(tensor) => B::dequantize(tensor),
            Self::Float(tensor) => tensor,
        }
    }
}

/// A type-level representation of the kind of a tensor.
pub trait TensorKind<B: Backend>: Clone + core::fmt::Debug {
    /// The primitive type of the tensor.
    type Primitive: Clone + core::fmt::Debug + Send;

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
