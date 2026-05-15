use burn_backend::TensorMetadata;
pub use burn_dispatch::DispatchTensor;
use burn_std::DType;

use crate::{
    Tensor,
    kind::Basic,
    ops::{BridgeTensor, TensorKindId},
};

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
    /// (e.g. passing [`BridgeTensor::Int`] when `K` is [`Float`](crate::Float).
    pub fn from_bridge(tensor: BridgeTensor) -> Self {
        let dtype = tensor.dtype();
        match (&tensor, K::id()) {
            (BridgeTensor::Bool(_), TensorKindId::Bool) if dtype.is_bool() => Self::new(tensor),
            (BridgeTensor::Int(_), TensorKindId::Int) if dtype.is_int() || dtype.is_uint() => {
                Self::new(tensor)
            }
            (BridgeTensor::Float(_), TensorKindId::Float) if dtype.is_float() => Self::new(tensor),
            (BridgeTensor::QFloat(_), TensorKindId::Float) if matches!(dtype, DType::QFloat(_)) => {
                Self::new(tensor)
            }
            (_, kind) => panic!("Expected kind {kind:?}, got dtype {dtype:?}"),
        }
    }

    /// Converts from a primitive tensor into a tensor.
    ///
    /// # Panics
    /// Panis if the primitive dtype does not match the tensor kind `K`.
    pub fn from_primitive(tensor: DispatchTensor) -> Self {
        match (tensor.dtype(), K::id()) {
            (DType::QFloat(_), TensorKindId::Float) => Self::new(BridgeTensor::QFloat(tensor)),
            (dtype, TensorKindId::Float) if dtype.is_float() => {
                Self::new(BridgeTensor::Float(tensor))
            }
            (dtype, TensorKindId::Int) if dtype.is_int() || dtype.is_uint() => {
                Self::new(BridgeTensor::Int(tensor))
            }
            (dtype, TensorKindId::Bool) if dtype.is_bool() => Self::new(BridgeTensor::Bool(tensor)),
            (dtype, kind) => panic!("Expected kind {kind:?}, got dtype {dtype:?}"),
        }
    }

    /// Converts the tensor into a primitive tensor.
    pub fn into_primitive(self) -> DispatchTensor {
        self.primitive.into()
    }
}

#[cfg(test)]
mod tests {
    use crate::{Bool, Int};

    use super::*;

    // -- into_bridge / from_bridge roundtrip --

    #[test]
    fn float_tensor_bridge_roundtrip() {
        let tensor = Tensor::<2>::zeros([2, 3], &Default::default());
        let shape = tensor.shape();
        let bridge = tensor.into_bridge();
        assert!(matches!(bridge, BridgeTensor::Float(_)));
        let tensor = Tensor::<2>::from_bridge(bridge);
        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn int_tensor_bridge_roundtrip() {
        let tensor = Tensor::<2, Int>::zeros([2, 3], &Default::default());
        let shape = tensor.shape();
        let bridge = tensor.into_bridge();
        assert!(matches!(bridge, BridgeTensor::Int(_)));
        let tensor = Tensor::<2, Int>::from_bridge(bridge);
        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn bool_tensor_bridge_roundtrip() {
        let tensor = Tensor::<2, Bool>::empty([2, 3], &Default::default());
        let shape = tensor.shape();
        let bridge = tensor.into_bridge();
        assert!(matches!(bridge, BridgeTensor::Bool(_)));
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
        // Construct a BridgeTensor::QFloat wrapping a non-qfloat dispatch tensor
        // kind tag says Float but dtype says otherwise.
        let inner = Tensor::<2, Int>::zeros([2, 3], &Default::default()).into_primitive();
        let bridge = BridgeTensor::QFloat(inner);
        let _tensor = Tensor::<2>::from_bridge(bridge);
    }

    // -- into_primitive / from_primitive roundtrip --

    #[test]
    fn float_primitive_roundtrip() {
        let tensor = Tensor::<2>::zeros([2, 3], &Default::default());
        let shape = tensor.shape();
        let primitive = tensor.into_primitive();
        let tensor = Tensor::<2>::from_primitive(primitive);
        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn int_primitive_roundtrip() {
        let tensor = Tensor::<2, Int>::zeros([2, 3], &Default::default());
        let shape = tensor.shape();
        let primitive = tensor.into_primitive();
        let tensor = Tensor::<2, Int>::from_primitive(primitive);
        assert_eq!(tensor.shape(), shape);
    }

    #[test]
    fn bool_primitive_roundtrip() {
        let tensor = Tensor::<2, Bool>::empty([2, 3], &Default::default());
        let shape = tensor.shape();
        let primitive = tensor.into_primitive();
        let tensor = Tensor::<2, Bool>::from_primitive(primitive);
        assert_eq!(tensor.shape(), shape);
    }

    // -- from_primitive panics on dtype/kind mismatch --

    #[test]
    #[should_panic(expected = "Expected kind Float")]
    fn from_primitive_int_dtype_as_float_panics() {
        let primitive = Tensor::<2, Int>::zeros([2, 3], &Default::default()).into_primitive();
        let _tensor = Tensor::<2>::from_primitive(primitive);
    }

    #[test]
    #[should_panic(expected = "Expected kind Int")]
    fn from_primitive_float_dtype_as_int_panics() {
        let primitive = Tensor::<2>::zeros([2, 3], &Default::default()).into_primitive();
        let _tensor = Tensor::<2, Int>::from_primitive(primitive);
    }

    #[test]
    #[should_panic(expected = "Expected kind Bool")]
    fn from_primitive_float_dtype_as_bool_panics() {
        let primitive = Tensor::<2>::zeros([2, 3], &Default::default()).into_primitive();
        let _tensor = Tensor::<2, Bool>::from_primitive(primitive);
    }
}
