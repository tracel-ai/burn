use crate::{DType, Shape, backend::Backend};

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

impl<B: Backend> TensorMetadata for TensorPrimitive<B> {
    fn dtype(&self) -> DType {
        match self {
            TensorPrimitive::Float(tensor) => tensor.dtype(),
            TensorPrimitive::QFloat(tensor) => tensor.dtype(),
        }
    }

    fn shape(&self) -> Shape {
        match self {
            TensorPrimitive::Float(tensor) => tensor.shape(),
            TensorPrimitive::QFloat(tensor) => tensor.shape(),
        }
    }

    fn rank(&self) -> usize {
        match self {
            TensorPrimitive::Float(tensor) => tensor.rank(),
            TensorPrimitive::QFloat(tensor) => tensor.rank(),
        }
    }
}

/// Tensor metadata trait for tensor primitive.
pub trait TensorMetadata: Clone + Send + Sync + core::fmt::Debug {
    /// The dtype of the tensor.
    fn dtype(&self) -> DType;
    /// The shape of the tensor.
    fn shape(&self) -> Shape;

    /// The number of dimensions of the tensor.
    fn rank(&self) -> usize {
        self.shape().num_dims()
    }
}

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
