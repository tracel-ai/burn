use std::mem::MaybeUninit;

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
#[derive(Debug)]
pub struct BridgeTensor {
    blob: Blob,
}

type InnerType = MaybeUninit<BridgeTensorVariant>;
type Blob = [u8; size_of::<InnerType>()];

impl Clone for BridgeTensor {
    fn clone(&self) -> Self {
        let inner = self.as_variant().clone();
        Self::new(inner)
    }
}

impl Drop for BridgeTensor {
    fn drop(&mut self) {
        unsafe {
            let blob = core::ptr::read(&self.blob);
            let mut inner: InnerType = core::mem::transmute(blob);
            inner.assume_init_drop();
        }
    }
}

impl BridgeTensor {
    pub fn as_variant(&self) -> &BridgeTensorVariant {
        unsafe {
            let tensor: &InnerType = core::mem::transmute(&self.blob);
            tensor.assume_init_ref()
        }
    }
    pub fn into_variant(self) -> BridgeTensorVariant {
        unsafe {
            let tensor: InnerType = core::mem::transmute(self.blob);
            tensor.assume_init()
        }
    }
    fn new(inner: BridgeTensorVariant) -> Self {
        let inner: Blob = unsafe { core::mem::transmute(inner) };

        Self { blob: inner }
    }
}

#[derive(Clone, Debug)]
/// Private type obfucated by Blob.
enum BridgeTensorVariant {
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
        match self.as_variant() {
            BridgeTensorVariant::Bool(tensor) => tensor.dtype(),
            BridgeTensorVariant::Int(tensor) => tensor.dtype(),
            BridgeTensorVariant::Float(tensor) => tensor.dtype(),
            BridgeTensorVariant::QFloat(tensor) => tensor.dtype(),
        }
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> burn_std::Shape {
        match self.as_variant() {
            BridgeTensorVariant::Bool(tensor) => tensor.shape(),
            BridgeTensorVariant::Int(tensor) => tensor.shape(),
            BridgeTensorVariant::Float(tensor) => tensor.shape(),
            BridgeTensorVariant::QFloat(tensor) => tensor.shape(),
        }
    }

    /// Returns the number of dimensions of the tensor.
    pub fn rank(&self) -> usize {
        match self.as_variant() {
            BridgeTensorVariant::Bool(tensor) => tensor.rank(),
            BridgeTensorVariant::Int(tensor) => tensor.rank(),
            BridgeTensorVariant::Float(tensor) => tensor.rank(),
            BridgeTensorVariant::QFloat(tensor) => tensor.rank(),
        }
    }

    pub(crate) fn as_dispatch(&self) -> &DispatchTensor {
        match self.as_variant() {
            BridgeTensorVariant::Bool(tensor) => tensor,
            BridgeTensorVariant::Int(tensor) => tensor,
            BridgeTensorVariant::Float(tensor) => tensor,
            BridgeTensorVariant::QFloat(tensor) => tensor,
        }
    }

    #[cfg(feature = "autodiff")]
    pub(crate) fn as_float(&self) -> &DispatchTensor {
        match self.as_variant() {
            BridgeTensorVariant::Float(tensor) => tensor,
            _ => panic!("Should be Float primitive kind"),
        }
    }

    pub(crate) fn into_dispatch_vec(tensors: Vec<Self>) -> Vec<DispatchTensor> {
        tensors.into_iter().map(Into::into).collect()
    }

    pub(crate) fn into_float(self) -> DispatchTensor {
        match self.into_variant() {
            BridgeTensorVariant::Float(tensor) => tensor,
            // Returns the dequantized float tensor.
            BridgeTensorVariant::QFloat(tensor) => {
                TensorPrimitive::<Dispatch>::QFloat(tensor).tensor()
            }
            _ => panic!("Should be Float primitive kind"),
        }
    }
}

impl From<BridgeTensor> for DispatchTensor {
    fn from(value: BridgeTensor) -> Self {
        match value.into_variant() {
            BridgeTensorVariant::Bool(tensor) => tensor,
            BridgeTensorVariant::Int(tensor) => tensor,
            BridgeTensorVariant::Float(tensor) => tensor,
            BridgeTensorVariant::QFloat(tensor) => tensor,
        }
    }
}
