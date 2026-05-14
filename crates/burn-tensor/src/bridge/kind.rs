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
    /// The name of the tensor kind.
    fn name() -> &'static str;
}

impl TensorKind for Float {
    fn name() -> &'static str {
        "Float"
    }
}

impl TensorKind for Int {
    fn name() -> &'static str {
        "Int"
    }
}

impl TensorKind for Bool {
    fn name() -> &'static str {
        "Bool"
    }
}

// TODO: maybe pub(crate)?
#[allow(missing_docs)]
#[derive(Clone, Debug)]
pub enum PrimitiveKind {
    Bool(DispatchTensor),
    Int(DispatchTensor),
    Float(DispatchTensor),
    QFloat(DispatchTensor),
}

impl PrimitiveKind {
    pub fn dtype(&self) -> burn_std::DType {
        match self {
            Self::Bool(tensor) => tensor.dtype(),
            Self::Int(tensor) => tensor.dtype(),
            Self::Float(tensor) => tensor.dtype(),
            Self::QFloat(tensor) => tensor.dtype(),
        }
    }

    pub fn shape(&self) -> burn_std::Shape {
        match self {
            Self::Bool(tensor) => tensor.shape(),
            Self::Int(tensor) => tensor.shape(),
            Self::Float(tensor) => tensor.shape(),
            Self::QFloat(tensor) => tensor.shape(),
        }
    }

    pub fn rank(&self) -> usize {
        match self {
            Self::Bool(tensor) => tensor.rank(),
            Self::Int(tensor) => tensor.rank(),
            Self::Float(tensor) => tensor.rank(),
            Self::QFloat(tensor) => tensor.rank(),
        }
    }

    pub(crate) fn as_dispatch(&self) -> &DispatchTensor {
        match self {
            PrimitiveKind::Bool(tensor) => tensor,
            PrimitiveKind::Int(tensor) => tensor,
            PrimitiveKind::Float(tensor) => tensor,
            PrimitiveKind::QFloat(tensor) => tensor,
        }
    }

    pub(crate) fn as_float(&self) -> &DispatchTensor {
        match self {
            PrimitiveKind::Float(tensor) => tensor,
            _ => panic!("Should be Float primitive kind"),
        }
    }

    pub(crate) fn into_dispatch_vec(tensors: Vec<Self>) -> Vec<DispatchTensor> {
        tensors.into_iter().map(Into::into).collect()
    }

    pub(crate) fn into_float(self) -> DispatchTensor {
        match self {
            PrimitiveKind::Float(tensor) => tensor,
            // Returns the dequantized float tensor.
            PrimitiveKind::QFloat(tensor) => TensorPrimitive::<Dispatch>::QFloat(tensor).tensor(),
            _ => panic!("Should be Float primitive kind"),
        }
    }
}

impl From<PrimitiveKind> for DispatchTensor {
    fn from(value: PrimitiveKind) -> Self {
        match value {
            PrimitiveKind::Bool(tensor) => tensor,
            PrimitiveKind::Int(tensor) => tensor,
            PrimitiveKind::Float(tensor) => tensor,
            PrimitiveKind::QFloat(tensor) => tensor,
        }
    }
}
