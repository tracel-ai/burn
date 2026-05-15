use alloc::vec::Vec;
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
pub trait TensorKind: Clone + Send + Sync + core::fmt::Debug {
    /// The name of the tensor kind.
    fn name() -> &'static str {
        Self::id().as_str()
    }

    /// The tensor kind identifier.
    fn id() -> TensorKindId;
}

impl TensorKind for Float {
    fn id() -> TensorKindId {
        TensorKindId::Float
    }
}

impl TensorKind for Int {
    fn id() -> TensorKindId {
        TensorKindId::Int
    }
}

impl TensorKind for Bool {
    fn id() -> TensorKindId {
        TensorKindId::Bool
    }
}

/// Runtime identifier for a tensor kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorKindId {
    /// A float tensor kind.
    Float,
    /// An integer tensor kind.
    Int,
    /// A boolean tensor kind.
    Bool,
}

impl TensorKindId {
    /// Get the string representation of the [`TensorKindId`].
    pub fn as_str(&self) -> &'static str {
        match self {
            TensorKindId::Float => "Float",
            TensorKindId::Int => "Int",
            TensorKindId::Bool => "Bool",
        }
    }
}

/// A type-tagged tensor at the bridge layer between the high-level tensor API
/// and the dispatch system.
///
/// `BridgeTensor` serves as the runtime representation for the public tensor
/// kinds (Float, Int, Bool) and internal variants like quantized floats, wrapping
/// the uniform [`DispatchTensor`] used by the underlying dispatch layer. This
/// separation keeps tensor kind tracking out of the backends while avoiding
/// exposure of backend-level primitives in the public API.
#[derive(Clone, Debug)]
pub enum BridgeTensor {
    /// A boolean tensor.
    Bool(DispatchTensor),
    /// An integer tensor.
    Int(DispatchTensor),
    /// A floating-point tensor.
    Float(DispatchTensor),
    /// A quantized floating-point tensor.
    QFloat(DispatchTensor),
}

impl BridgeTensor {
    /// Returns the dtype of the tensor.
    pub fn dtype(&self) -> burn_std::DType {
        match self {
            Self::Bool(tensor) => tensor.dtype(),
            Self::Int(tensor) => tensor.dtype(),
            Self::Float(tensor) => tensor.dtype(),
            Self::QFloat(tensor) => tensor.dtype(),
        }
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> burn_std::Shape {
        match self {
            Self::Bool(tensor) => tensor.shape(),
            Self::Int(tensor) => tensor.shape(),
            Self::Float(tensor) => tensor.shape(),
            Self::QFloat(tensor) => tensor.shape(),
        }
    }

    /// Returns the number of dimensions of the tensor.
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
            BridgeTensor::Bool(tensor) => tensor,
            BridgeTensor::Int(tensor) => tensor,
            BridgeTensor::Float(tensor) => tensor,
            BridgeTensor::QFloat(tensor) => tensor,
        }
    }

    #[cfg(feature = "autodiff")]
    pub(crate) fn as_float(&self) -> &DispatchTensor {
        match self {
            BridgeTensor::Float(tensor) => tensor,
            _ => panic!("Should be Float primitive kind"),
        }
    }

    pub(crate) fn into_dispatch_vec(tensors: Vec<Self>) -> Vec<DispatchTensor> {
        tensors.into_iter().map(Into::into).collect()
    }

    pub(crate) fn into_float(self) -> DispatchTensor {
        match self {
            BridgeTensor::Float(tensor) => tensor,
            // Returns the dequantized float tensor.
            BridgeTensor::QFloat(tensor) => TensorPrimitive::<Dispatch>::QFloat(tensor).tensor(),
            _ => panic!("Should be Float primitive kind"),
        }
    }
}

impl From<BridgeTensor> for DispatchTensor {
    fn from(value: BridgeTensor) -> Self {
        match value {
            BridgeTensor::Bool(tensor) => tensor,
            BridgeTensor::Int(tensor) => tensor,
            BridgeTensor::Float(tensor) => tensor,
            BridgeTensor::QFloat(tensor) => tensor,
        }
    }
}
