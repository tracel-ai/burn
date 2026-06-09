use burn_backend::{Backend, TensorMetadata};
pub use burn_dispatch::DispatchTensor;
use burn_dispatch::{BackendTensor, DispatchKindConversion};
use burn_std::DType;

use crate::{
    Bool, Float, Int, Tensor,
    kind::Basic,
    ops::{BridgeKind, BridgeTensor, Kind},
};

use alloc::{format, string::String};

impl<const D: usize, K> Tensor<D, K>
where
    K: Basic,
{
    /// Converts the tensor into its bridge-layer representation.
    ///
    /// This is primarily intended for backend extensions, allowing custom operations
    /// to be inserted between the high-level tensor API and the dispatch layer before
    /// deferring to a concrete backend.
    pub fn into_bridge(self) -> BridgeTensor {
        self.primitive
    }

    /// Reconstructs a tensor from its [`BridgeTensor`] bridge representation.
    ///
    /// This is the inverse of [`Tensor::into_bridge`] and is primarily intended
    /// for backend extensions, to wrap the output of a custom operation back into
    /// the high-level tensor API.
    ///
    /// # Panics
    ///
    /// Panics if the [`BridgeTensor`] variant does not match the tensor kind `K`
    /// (e.g. passing an int [`BridgeTensor`] when `K` is [`Float`](crate::Float).
    pub fn from_bridge(tensor: BridgeTensor) -> Self {
        let dtype = tensor.dtype();
        match (tensor.kind(), K::KIND) {
            (BridgeKind::Bool, Kind::Bool) if dtype.is_bool() => Self::new(tensor),
            (BridgeKind::Int, Kind::Int) if dtype.is_int() || dtype.is_uint() => Self::new(tensor),
            (BridgeKind::Float, Kind::Float) if dtype.is_float() => Self::new(tensor),
            (BridgeKind::QFloat, Kind::Float) if matches!(dtype, DType::QFloat(_)) => {
                Self::new(tensor)
            }
            (_, kind) => panic!("Expected kind {kind:?}, got dtype {dtype:?}"),
        }
    }

    /// Converts from a dispatch tensor into a tensor.
    ///
    /// # Panics
    /// Panis if the dispatch dtype does not match the tensor kind `K`.
    pub fn from_dispatch(tensor: DispatchTensor) -> Self {
        match (tensor.dtype(), K::KIND) {
            (DType::QFloat(_), Kind::Float) => Self::new(BridgeTensor::qfloat(tensor)),
            (dtype, Kind::Float) if dtype.is_float() => Self::new(BridgeTensor::float(tensor)),
            (dtype, Kind::Int) if dtype.is_int() || dtype.is_uint() => {
                Self::new(BridgeTensor::int(tensor))
            }
            (dtype, Kind::Bool) if dtype.is_bool() => Self::new(BridgeTensor::bool(tensor)),
            (dtype, kind) => panic!("Expected kind {kind:?}, got dtype {dtype:?}"),
        }
    }

    /// Converts the tensor into a dispatch tensor.
    pub fn into_dispatch(self) -> DispatchTensor {
        self.primitive.into()
    }

    /// Safely downcasts a [`Tensor`] into its low-level backend primitive.
    ///
    /// This is primarily intended for backend extensions where interfacing with the direct backend
    /// primitive (e.g., `B::FloatTensorPrimitive` or `AutodiffTensor<B>`) is necessary.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Extract the underlying CubeCL tensor primitive on the `Wgpu` backend
    /// let cube_tensor = tensor.try_into_primitive::<Wgpu>()?;
    ///
    /// // For an autodiff tensor, we can get the wrapped CubeCL tensor primitive on the `Autodiff<Wgpu>` backend
    /// let ad_tensor = tensor.try_into_primitive::<Autodiff<Wgpu>>()?;
    ///```
    ///
    /// # Errors
    ///
    /// Returns a [`PrimitiveConversionError`] if the tensor does not currently live on the requested
    /// backend `B` (including `Autodiff<B>` mismatch).
    ///
    /// ```
    pub fn try_into_primitive<B: Backend>(
        self,
    ) -> Result<<K as BackendPrimitive<B>>::Primitive, PrimitiveConversionError>
    where
        K: BackendPrimitive<B>,
        DispatchTensor: DispatchKindConversion<B>,
    {
        let dispatch = self.primitive.into();
        // `tensor.try_into_backend::<Autodiff<B>>()` returns a `BackendTensor::Float(AutodiffTensor<B>)`
        // so it is automatically handled by the `try_into_primitive` impl for `Float`
        let tensor = <DispatchTensor as DispatchKindConversion<B>>::try_into_backend(dispatch)
            .map_err(PrimitiveConversionError::BackendMismatch)?;
        <K as BackendPrimitive<B>>::try_into_primitive(tensor)
    }

    /// Reconstructs a high-level [`Tensor`] from a concrete backend primitive.
    ///
    /// This is the inverse of [`Tensor::try_into_primitive`].
    ///
    /// # Panics
    /// Panis if the tensor kind `K` does not match the tensor underlying primitive kind.
    pub fn from_primitive<B: Backend>(primitive: <K as BackendPrimitive<B>>::Primitive) -> Self
    where
        K: BackendPrimitive<B>,
        DispatchTensor: DispatchKindConversion<B>,
    {
        let tensor = <K as BackendPrimitive<B>>::from_primitive(primitive);
        let dispatch = <DispatchTensor as DispatchKindConversion<B>>::from_backend(tensor);
        Self::from_dispatch(dispatch)
    }
}

/// Error returned when a [`DispatchTensor`] cannot be converted to the requested concrete primitive type.
#[derive(Debug, PartialEq)]
pub enum PrimitiveConversionError {
    /// The dispatch tensor's backend variant does not match the requested backend.
    ///
    /// For example, extracting a `Wgpu` primitive from a `Cuda` dispatch tensor.
    BackendMismatch(String),
    /// The tensor kind does not match the requested primitive kind.
    ///
    /// For example, extracting a float primitive from an int tensor.
    KindMismatch(String),
}

/// Trait to safely extract and wrap backend-specific primitives based on the high-level tensor kind.
///
/// This trait functions as a type-level map linking frontend kinds (`Float`, `Int`, `Bool`)
/// to their corresponding backend-associated types (`B::FloatTensorPrimitive`, `B::IntTensorPrimitive`, etc.).
pub trait BackendPrimitive<B: Backend> {
    /// The backend tensor primitive.
    type Primitive;

    /// Attempts to unpack the type-erased [`BackendTensor`] enum wrapper into the concrete
    /// variant expected by this kind.
    ///
    /// # Errors
    ///
    /// Returns an error if there is a variant type mismatch (e.g., attempting to extract an
    /// `Int` primitive out of a `BackendTensor::Float` enum variant).
    fn try_into_primitive(
        tensor: BackendTensor<B>,
    ) -> Result<Self::Primitive, PrimitiveConversionError>;

    /// Wraps a backend tensor primitive into its corresponding `BackendTensor` enum variant.
    fn from_primitive(primitive: Self::Primitive) -> BackendTensor<B>;
}

impl<B: Backend> BackendPrimitive<B> for Float {
    // NOTE: not implemented for QFloat
    type Primitive = B::FloatTensorPrimitive;

    fn try_into_primitive(
        tensor: BackendTensor<B>,
    ) -> Result<Self::Primitive, PrimitiveConversionError> {
        match tensor {
            BackendTensor::Float(t) => Ok(t),
            other => Err(PrimitiveConversionError::KindMismatch(format!(
                "Expected Float primitive, got variant: {}",
                other.name()
            ))),
        }
    }

    fn from_primitive(primitive: Self::Primitive) -> BackendTensor<B> {
        BackendTensor::Float(primitive)
    }
}

impl<B: Backend> BackendPrimitive<B> for Int {
    type Primitive = B::IntTensorPrimitive;

    fn try_into_primitive(
        tensor: BackendTensor<B>,
    ) -> Result<Self::Primitive, PrimitiveConversionError> {
        match tensor {
            BackendTensor::Int(t) => Ok(t),
            other => Err(PrimitiveConversionError::KindMismatch(format!(
                "Expected Int primitive, got variant: {}",
                other.name()
            ))),
        }
    }

    fn from_primitive(primitive: Self::Primitive) -> BackendTensor<B> {
        BackendTensor::Int(primitive)
    }
}

impl<B: Backend> BackendPrimitive<B> for Bool {
    type Primitive = B::BoolTensorPrimitive;

    fn try_into_primitive(
        tensor: BackendTensor<B>,
    ) -> Result<Self::Primitive, PrimitiveConversionError> {
        match tensor {
            BackendTensor::Bool(t) => Ok(t),
            other => Err(PrimitiveConversionError::KindMismatch(format!(
                "Expected Bool primitive, got variant: {}",
                other.name()
            ))),
        }
    }

    fn from_primitive(primitive: Self::Primitive) -> BackendTensor<B> {
        BackendTensor::Bool(primitive)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Bool, Int};

    use super::*;

    type TestBackend = burn_dispatch::backends::Flex;
    type TestAutodiffBackend = burn_dispatch::backends::Autodiff<burn_dispatch::backends::Flex>;

    // -- into_bridge / from_bridge roundtrip --

    #[test]
    fn float_tensor_bridge_roundtrip() {
        let tensor = Tensor::<2>::zeros([2, 3], &Default::default());
        let shape = tensor.shape();
        let bridge = tensor.into_bridge();
        assert!(bridge.is_float());
        let tensor = Tensor::<2>::from_bridge(bridge);
        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn int_tensor_bridge_roundtrip() {
        let tensor = Tensor::<2, Int>::zeros([2, 3], &Default::default());
        let shape = tensor.shape();
        let bridge = tensor.into_bridge();
        assert!(bridge.is_int());
        let tensor = Tensor::<2, Int>::from_bridge(bridge);
        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn bool_tensor_bridge_roundtrip() {
        let tensor = Tensor::<2, Bool>::empty([2, 3], &Default::default());
        let shape = tensor.shape();
        let bridge = tensor.into_bridge();
        assert!(bridge.is_bool());
        let tensor = Tensor::<2, Bool>::from_bridge(bridge);
        assert_eq!(tensor.shape(), shape);
    }

    // -- from_bridge panics on kind mismatch --

    #[test]
    #[should_panic(expected = "Expected kind Float")]
    fn from_bridge_int_as_float_panics() {
        let bridge = Tensor::<2, Int>::zeros([2, 3], &Default::default()).into_bridge();
        let _tensor = Tensor::<2>::from_bridge(bridge);
    }

    #[test]
    #[should_panic(expected = "Expected kind Float")]
    fn from_bridge_bool_as_float_panics() {
        let bridge = Tensor::<2, Bool>::empty([2, 3], &Default::default()).into_bridge();
        let _tensor = Tensor::<2>::from_bridge(bridge);
    }

    #[test]
    #[should_panic(expected = "Expected kind Int")]
    fn from_bridge_float_as_int_panics() {
        let bridge = Tensor::<2>::zeros([2, 3], &Default::default()).into_bridge();
        let _tensor = Tensor::<2, Int>::from_bridge(bridge);
    }

    #[test]
    #[should_panic(expected = "Expected kind Bool")]
    fn from_bridge_int_as_bool_panics() {
        let bridge = Tensor::<2, Int>::zeros([2, 3], &Default::default()).into_bridge();
        let _tensor = Tensor::<2, Bool>::from_bridge(bridge);
    }

    #[test]
    #[should_panic(expected = "Expected kind Float")]
    fn from_bridge_qfloat_variant_with_int_dtype_panics() {
        // Construct a QFloat bridge tensor wrapping a non-qfloat dispatch tensor:
        // kind tag says Float but dtype says otherwise.
        let inner = Tensor::<2, Int>::zeros([2, 3], &Default::default()).into_dispatch();
        let bridge = BridgeTensor::qfloat(inner);
        let _tensor = Tensor::<2>::from_bridge(bridge);
    }

    // -- into_dispatch / from_dispatch roundtrip --

    #[test]
    fn float_primitive_roundtrip() {
        let tensor = Tensor::<2>::zeros([2, 3], &Default::default());
        let shape = tensor.shape();
        let primitive = tensor.into_dispatch();
        let tensor = Tensor::<2>::from_dispatch(primitive);
        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn int_primitive_roundtrip() {
        let tensor = Tensor::<2, Int>::zeros([2, 3], &Default::default());
        let shape = tensor.shape();
        let primitive = tensor.into_dispatch();
        let tensor = Tensor::<2, Int>::from_dispatch(primitive);
        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn bool_primitive_roundtrip() {
        let tensor = Tensor::<2, Bool>::empty([2, 3], &Default::default());
        let shape = tensor.shape();
        let primitive = tensor.into_dispatch();
        let tensor = Tensor::<2, Bool>::from_dispatch(primitive);
        assert_eq!(tensor.shape(), shape);
    }

    // -- from_dispatch panics on dtype/kind mismatch --

    #[test]
    #[should_panic(expected = "Expected kind Float")]
    fn from_dispatch_int_dtype_as_float_panics() {
        let primitive = Tensor::<2, Int>::zeros([2, 3], &Default::default()).into_dispatch();
        let _tensor = Tensor::<2>::from_dispatch(primitive);
    }

    #[test]
    #[should_panic(expected = "Expected kind Int")]
    fn from_dispatch_float_dtype_as_int_panics() {
        let primitive = Tensor::<2>::zeros([2, 3], &Default::default()).into_dispatch();
        let _tensor = Tensor::<2, Int>::from_dispatch(primitive);
    }

    #[test]
    #[should_panic(expected = "Expected kind Bool")]
    fn from_dispatch_float_dtype_as_bool_panics() {
        let primitive = Tensor::<2>::zeros([2, 3], &Default::default()).into_dispatch();
        let _tensor = Tensor::<2, Bool>::from_dispatch(primitive);
    }

    // -- try_into_primitive / from_primitive roundtrip --

    #[test]
    fn float_backend_primitive_roundtrip() {
        let device = Default::default();
        let tensor = Tensor::<2>::zeros([2, 3], &device);
        let shape = tensor.shape();

        let primitive = tensor
            .try_into_primitive::<TestBackend>()
            .expect("Failed to extract Float primitive");
        let tensor = Tensor::<2>::from_primitive::<TestBackend>(primitive);

        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn int_backend_primitive_roundtrip() {
        let device = Default::default();
        let tensor = Tensor::<2, Int>::zeros([2, 3], &device);
        let shape = tensor.shape();

        let primitive = tensor
            .try_into_primitive::<TestBackend>()
            .expect("Failed to extract Int primitive");
        let tensor = Tensor::<2, Int>::from_primitive::<TestBackend>(primitive);

        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn bool_backend_primitive_roundtrip() {
        let device = Default::default();
        let tensor = Tensor::<2, Bool>::empty([2, 3], &device);
        let shape = tensor.shape();

        let primitive = tensor
            .try_into_primitive::<TestBackend>()
            .expect("Failed to extract Bool primitive");
        let tensor = Tensor::<2, Bool>::from_primitive::<TestBackend>(primitive);

        assert_eq!(tensor.shape(), shape);
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn autodiff_backend_primitive_roundtrip() {
        let device = crate::Device::default().autodiff();
        let tensor = Tensor::<2, Float>::empty([2, 3], &device).require_grad();
        let require_grad = tensor.is_require_grad();

        let primitive = tensor
            .try_into_primitive::<TestAutodiffBackend>()
            .expect("Failed to extract Autodiff primitive");
        let tensor = Tensor::<2, Float>::from_primitive::<TestAutodiffBackend>(primitive);

        assert_eq!(tensor.is_require_grad(), require_grad);
    }

    // -- try_into_primitive panics on backend or kind mismatch --

    #[cfg(feature = "autodiff")]
    #[test]
    fn try_into_primitive_backend_mismatch() {
        let device = Default::default();
        let tensor = Tensor::<2>::zeros([2, 3], &device);

        let err = tensor
            .try_into_primitive::<TestAutodiffBackend>()
            .unwrap_err();

        assert!(matches!(err, PrimitiveConversionError::BackendMismatch(_)));
        assert!(format!("{err:?}").contains("Expected Autodiff tensor, got backend:"));
    }

    #[test]
    #[should_panic(expected = "Expected kind Float")]
    fn try_into_primitive_kind_mismatch() {
        let device = Default::default();
        let tensor = Tensor::<2, Int>::zeros([2, 3], &device);

        let primitive = tensor
            .try_into_primitive::<TestBackend>()
            .expect("Failed to extract Int primitive");
        let _tensor = Tensor::<2, Float>::from_primitive::<TestBackend>(primitive);
    }

    // -- BackendPrimitive trait error handling on mismatched variants --

    #[test]
    fn try_into_primitive_float_with_int_variant_returns_err() {
        let device = Default::default();

        // Valid Int primitive
        let int_tensor = Tensor::<2, Int>::zeros([2, 3], &device);
        let int_primitive = int_tensor.try_into_primitive::<TestBackend>().unwrap();

        // Wrap it artificially into the Backend layer
        let backend_tensor = BackendTensor::<TestBackend>::Int(int_primitive);

        // Attempt to extract it as a Float primitive using the BackendPrimitive trait
        let err = <Float as BackendPrimitive<TestBackend>>::try_into_primitive(backend_tensor)
            .unwrap_err();

        assert_eq!(
            err,
            PrimitiveConversionError::KindMismatch(
                "Expected Float primitive, got variant: Int".into()
            )
        );
    }
}
