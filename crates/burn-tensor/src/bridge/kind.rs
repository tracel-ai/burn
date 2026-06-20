use alloc::vec::Vec;
use burn_backend::{TensorMetadata, TensorPrimitive, get_device_settings};
use burn_dispatch::{Dispatch, DispatchTensor};
use burn_std::DeviceSettings;

/// A type-level representation of the kind of a float tensor
#[derive(Clone, Debug)]
pub struct Float;

/// A type-level representation of the kind of a int tensor.
#[derive(Clone, Debug)]
pub struct Int;

/// A type-level representation of the kind of a bool tensor.
#[derive(Clone, Debug)]
pub struct Bool;

/// A type-level representation of the kind of a complex tensor.
#[derive(Clone, Debug)]
pub struct Complex;

mod sealed {
    pub trait Sealed {}
}

impl sealed::Sealed for Float {}
impl sealed::Sealed for Int {}
impl sealed::Sealed for Bool {}
impl sealed::Sealed for Complex {}

/// A type-level representation of the kind of a tensor.
/// Metadata access is lazy.
///
/// # Notes
/// This trait is intentionally sealed to keep the set of tensor kinds closed.
///
/// Although exposed publicly, tensor kinds are not meant to be extensible:
/// the backend dispatch system, `DType`, and all tensor ops assume a fixed,
/// closed set of tensor kinds (e.g. Float, Int, Bool), each mapping directly to a
/// corresponding backend implementation.
pub trait TensorKind: sealed::Sealed + Clone + Send + Sync + core::fmt::Debug {
    /// The tensor kind identifier.
    const KIND: Kind;

    /// The name of the tensor kind.
    fn name() -> &'static str {
        Self::KIND.as_str()
    }
}

impl TensorKind for Float {
    const KIND: Kind = Kind::Float;
}

impl TensorKind for Int {
    const KIND: Kind = Kind::Int;
}

impl TensorKind for Bool {
    const KIND: Kind = Kind::Bool;
}

impl TensorKind for Complex {
    const KIND: Kind = Kind::Complex;
}

/// Represents the kind of a [`Tensor`](crate::Tensor).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A float tensor kind.
    Float,
    /// An integer tensor kind.
    Int,
    /// A boolean tensor kind.
    Bool,
    /// A complex tensor kind.
    Complex,
}

impl Kind {
    /// Get the string representation of the [`Kind`].
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Float => "Float",
            Kind::Int => "Int",
            Kind::Bool => "Bool",
            Kind::Complex => "Complex",
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
    blob: bridge_opaque::Opaque,
}

// Aligned, type-erased storage for `BridgeTensorVariant`. See `crate::macros`
// for why this indirection exists (it keeps the dispatch type tree out of
// downstream MIR).
burn_std::obfuscate!(
    type: BridgeTensorVariant,
    module: bridge_opaque,
    derives: [Send, Sync]
);

impl core::fmt::Debug for BridgeTensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BridgeTensor")
            .field("kind", &self.kind())
            .finish()
    }
}

impl Clone for BridgeTensor {
    fn clone(&self) -> Self {
        Self::new(self.as_variant().clone())
    }
}

impl BridgeTensor {
    fn as_variant(&self) -> &BridgeTensorVariant {
        self.blob.as_ref()
    }

    fn into_variant(self) -> BridgeTensorVariant {
        self.blob.into_inner()
    }

    fn new(inner: BridgeTensorVariant) -> Self {
        Self {
            blob: bridge_opaque::Opaque::new(inner),
        }
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
    /// A complex tensor.
    Complex(DispatchTensor),
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
    /// A complex tensor.
    Complex,
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
        /// Builds a bridge tensor that wraps a complex dispatch tensor.
        ///
        /// Available with the `extension` feature for backend-extension authors.
        fn complex(tensor: DispatchTensor) -> Self {
            Self::new(BridgeTensorVariant::Complex(tensor))
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
            BridgeTensorVariant::Complex(_) => BridgeKind::Complex,
        }
    }

    /// Returns `true` if this tensor is the float variant.
    pub fn is_float(&self) -> bool {
        matches!(self.kind(), BridgeKind::Float)
    }

    /// Returns `true` if this tensor is the int variant.
    pub fn is_int(&self) -> bool {
        matches!(self.kind(), BridgeKind::Int)
    }

    /// Returns `true` if this tensor is the bool variant.
    pub fn is_bool(&self) -> bool {
        matches!(self.kind(), BridgeKind::Bool)
    }

    /// Returns `true` if this tensor is the quantized float variant.
    pub fn is_qfloat(&self) -> bool {
        matches!(self.kind(), BridgeKind::QFloat)
    }
    /// Returns `true` if this tensor is the complex variant.
    pub fn is_complex(&self) -> bool {
        matches!(self.kind(), BridgeKind::Complex)
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
                BridgeTensorVariant::Complex(t) => (BridgeKind::Complex, t),
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
                BridgeTensorVariant::Complex(t) => (BridgeKind::Complex, t),
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
            BridgeTensorVariant::Complex(tensor) => tensor.dtype(),
        }
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> burn_std::Shape {
        match self.as_variant() {
            BridgeTensorVariant::Bool(tensor) => tensor.shape(),
            BridgeTensorVariant::Int(tensor) => tensor.shape(),
            BridgeTensorVariant::Float(tensor) => tensor.shape(),
            BridgeTensorVariant::QFloat(tensor) => tensor.shape(),
            BridgeTensorVariant::Complex(tensor) => tensor.shape(),
        }
    }

    /// Returns the number of dimensions of the tensor.
    pub fn rank(&self) -> usize {
        match self.as_variant() {
            BridgeTensorVariant::Bool(tensor) => tensor.rank(),
            BridgeTensorVariant::Int(tensor) => tensor.rank(),
            BridgeTensorVariant::Float(tensor) => tensor.rank(),
            BridgeTensorVariant::QFloat(tensor) => tensor.rank(),
            BridgeTensorVariant::Complex(tensor) => tensor.rank(),
        }
    }

    pub(crate) fn as_dispatch(&self) -> &DispatchTensor {
        match self.as_variant() {
            BridgeTensorVariant::Bool(tensor) => tensor,
            BridgeTensorVariant::Int(tensor) => tensor,
            BridgeTensorVariant::Float(tensor) => tensor,
            BridgeTensorVariant::QFloat(tensor) => tensor,
            BridgeTensorVariant::Complex(tensor) => tensor,
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

    pub(crate) fn device_settings(&self) -> DeviceSettings {
        let device = match self.as_variant() {
            BridgeTensorVariant::Bool(tensor) => tensor.device(),
            BridgeTensorVariant::Int(tensor) => tensor.device(),
            BridgeTensorVariant::Float(tensor) => tensor.device(),
            BridgeTensorVariant::QFloat(tensor) => tensor.device(),
            BridgeTensorVariant::Complex(tensor) => tensor.device(),
        };

        get_device_settings::<Dispatch>(&device)
    }

    pub(crate) fn into_complex(self) -> DispatchTensor {
        match self.into_variant() {
            BridgeTensorVariant::Complex(tensor) => tensor,
            _ => panic!("Should be Complex primitive kind"),
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
            BridgeTensorVariant::Complex(tensor) => tensor,
        }
    }
}
