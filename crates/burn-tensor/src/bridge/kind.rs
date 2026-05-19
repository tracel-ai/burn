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
pub struct BridgeTensor {
    blob: Blob,
}

impl core::fmt::Debug for BridgeTensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BridgeTensor")
            .field("kind", &self.kind())
            .finish()
    }
}

type InnerType = MaybeUninit<BridgeTensorVariant>;

/// Storage for [`BridgeTensor`]. Holds the raw bytes of an [`InnerType`] while
/// preserving its alignment via the zero-sized `_align` field, so the backing
/// memory can be safely reinterpreted as an [`InnerType`] reference.
#[repr(C)]
struct Blob {
    bytes: [u8; size_of::<InnerType>()],
    _align: [InnerType; 0],
}

impl Clone for BridgeTensor {
    fn clone(&self) -> Self {
        let inner = self.as_variant().clone();
        Self::new(inner)
    }
}

impl Drop for BridgeTensor {
    fn drop(&mut self) {
        unsafe {
            let inner: &mut InnerType = &mut *(self.blob.bytes.as_mut_ptr() as *mut InnerType);
            inner.assume_init_drop();
        }
    }
}

impl BridgeTensor {
    fn as_variant(&self) -> &BridgeTensorVariant {
        unsafe {
            let tensor: &InnerType = &*(self.blob.bytes.as_ptr() as *const InnerType);
            tensor.assume_init_ref()
        }
    }
    fn into_variant(mut self) -> BridgeTensorVariant {
        unsafe {
            let inner: InnerType =
                core::ptr::read(self.blob.bytes.as_mut_ptr() as *const InnerType);
            core::mem::forget(self);
            inner.assume_init()
        }
    }
    fn new(inner: BridgeTensorVariant) -> Self {
        let mut blob = Blob {
            bytes: [0u8; size_of::<InnerType>()],
            _align: [],
        };
        unsafe {
            let dst = blob.bytes.as_mut_ptr() as *mut InnerType;
            dst.write(MaybeUninit::new(inner));
        }
        Self { blob }
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

/// Runtime tag identifying which variant a [`BridgeTensor`] wraps.
///
/// Exposed so callers can dispatch on the variant without having to reach the
/// private [`BridgeTensorVariant`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BridgeKind {
    /// A boolean tensor.
    Bool,
    /// An integer tensor.
    Int,
    /// A floating-point tensor.
    Float,
    /// A quantized floating-point tensor.
    QFloat,
}

/// Switches visibility based on the `extension` feature: `pub` when enabled
/// (so backend-extension authors can call these), `pub(crate)` otherwise (so
/// burn-tensor's own ops keep working without leaking the dispatch types).
macro_rules! ext_fn {
    ($(#[$meta:meta])* fn $($tt:tt)+) => {
        #[cfg(feature = "extension")]
        $(#[$meta])*
        pub fn $($tt)+
        #[cfg(not(feature = "extension"))]
        $(#[$meta])*
        pub(crate) fn $($tt)+
    };
}

impl BridgeTensor {
    ext_fn! {
        /// Builds a bridge tensor that wraps a floating-point dispatch tensor.
        ///
        /// Available with the `extension` feature for backend-extension authors.
        fn float(tensor: DispatchTensor) -> Self {
            Self::new(BridgeTensorVariant::Float(tensor))
        }
    }

    ext_fn! {
        /// Builds a bridge tensor that wraps an integer dispatch tensor.
        ///
        /// Available with the `extension` feature for backend-extension authors.
        fn int(tensor: DispatchTensor) -> Self {
            Self::new(BridgeTensorVariant::Int(tensor))
        }
    }

    ext_fn! {
        /// Builds a bridge tensor that wraps a boolean dispatch tensor.
        ///
        /// Available with the `extension` feature for backend-extension authors.
        fn bool(tensor: DispatchTensor) -> Self {
            Self::new(BridgeTensorVariant::Bool(tensor))
        }
    }

    ext_fn! {
        /// Builds a bridge tensor that wraps a quantized floating-point dispatch tensor.
        ///
        /// Available with the `extension` feature for backend-extension authors.
        fn qfloat(tensor: DispatchTensor) -> Self {
            Self::new(BridgeTensorVariant::QFloat(tensor))
        }
    }

    /// Returns the runtime tag identifying which variant this tensor wraps.
    pub fn kind(&self) -> BridgeKind {
        match self.as_variant() {
            BridgeTensorVariant::Bool(_) => BridgeKind::Bool,
            BridgeTensorVariant::Int(_) => BridgeKind::Int,
            BridgeTensorVariant::Float(_) => BridgeKind::Float,
            BridgeTensorVariant::QFloat(_) => BridgeKind::QFloat,
        }
    }

    /// Returns `true` if this tensor is the [`BridgeKind::Float`] variant.
    pub fn is_float(&self) -> bool {
        matches!(self.kind(), BridgeKind::Float)
    }

    /// Returns `true` if this tensor is the [`BridgeKind::Int`] variant.
    pub fn is_int(&self) -> bool {
        matches!(self.kind(), BridgeKind::Int)
    }

    /// Returns `true` if this tensor is the [`BridgeKind::Bool`] variant.
    pub fn is_bool(&self) -> bool {
        matches!(self.kind(), BridgeKind::Bool)
    }

    /// Returns `true` if this tensor is the [`BridgeKind::QFloat`] variant.
    pub fn is_qfloat(&self) -> bool {
        matches!(self.kind(), BridgeKind::QFloat)
    }

    ext_fn! {
        /// Consumes the bridge tensor and returns its variant tag together with
        /// the underlying dispatch tensor.
        ///
        /// Available with the `extension` feature for backend-extension authors.
        fn into_parts(self) -> (BridgeKind, DispatchTensor) {
            match self.into_variant() {
                BridgeTensorVariant::Bool(t) => (BridgeKind::Bool, t),
                BridgeTensorVariant::Int(t) => (BridgeKind::Int, t),
                BridgeTensorVariant::Float(t) => (BridgeKind::Float, t),
                BridgeTensorVariant::QFloat(t) => (BridgeKind::QFloat, t),
            }
        }
    }

    ext_fn! {
        /// Borrows the bridge tensor as its variant tag together with a reference
        /// to the underlying dispatch tensor.
        ///
        /// Available with the `extension` feature for backend-extension authors.
        fn as_parts(&self) -> (BridgeKind, &DispatchTensor) {
            match self.as_variant() {
                BridgeTensorVariant::Bool(t) => (BridgeKind::Bool, t),
                BridgeTensorVariant::Int(t) => (BridgeKind::Int, t),
                BridgeTensorVariant::Float(t) => (BridgeKind::Float, t),
                BridgeTensorVariant::QFloat(t) => (BridgeKind::QFloat, t),
            }
        }
    }

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
