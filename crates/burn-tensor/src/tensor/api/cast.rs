use burn_backend::ops::{BoolTensorOps, FloatTensorOps, IntTensorOps};
use burn_backend::{DType, FloatDType, IntDType};
use burn_dispatch::Dispatch;

use crate::kind::Basic;
use crate::ops::BridgeTensor;
use crate::{Bool, Float, Int, Tensor};

/// Trait for types that represent a valid cast target from a tensor of kind `K`.
///
/// The generic parameter `K` is the *input* tensor kind ([`Float`], [`Int`], or [`Bool`]).
/// Implementors declare the output kind and provide the actual cast logic.
pub trait Cast<const D: usize, K: Basic> {
    /// The output tensor kind after casting.
    type OutputKind: Basic;

    /// Cast a tensor primitive to the target dtype.
    fn cast(tensor: Tensor<D, K>, dtype: Self) -> Tensor<D, Self::OutputKind>;
}

// --- Float input impls ---

impl<const D: usize> Cast<D, Float> for FloatDType {
    type OutputKind = Float;

    fn cast(tensor: Tensor<D, Float>, dtype: Self) -> Tensor<D, Float> {
        if let BridgeTensor::Float(_) = tensor.primitive {
            let current: FloatDType = tensor.dtype().into();
            if current == dtype {
                return tensor;
            }
            Tensor::new(BridgeTensor::Float(Dispatch::float_cast(
                tensor.primitive.into_float(),
                dtype,
            )))
        } else {
            panic!("Should be Float primitive kind");
        }
    }
}

impl<const D: usize> Cast<D, Float> for IntDType {
    type OutputKind = Int;

    fn cast(tensor: Tensor<D, Float>, dtype: Self) -> Tensor<D, Int> {
        Tensor::new(BridgeTensor::Int(Dispatch::float_into_int(
            tensor.primitive.into_float(),
            dtype,
        )))
    }
}

/// Backward-compatible impl: only float `DType` variants are accepted.
///
/// # Panics
///
/// Panics if `dtype` is not a float variant (e.g., `DType::I32`).
/// Use [`IntDType`] directly for cross-kind casting to int.
impl<const D: usize> Cast<D, Float> for DType {
    type OutputKind = Float;

    fn cast(tensor: Tensor<D, Float>, dtype: Self) -> Tensor<D, Float> {
        let float_dtype: FloatDType = dtype.into();
        <FloatDType as Cast<D, Float>>::cast(tensor, float_dtype)
    }
}

// --- Int input impls ---

impl<const D: usize> Cast<D, Int> for IntDType {
    type OutputKind = Int;

    fn cast(tensor: Tensor<D, Int>, dtype: Self) -> Tensor<D, Int> {
        let current: IntDType = tensor.primitive.dtype().into();
        if current == dtype {
            return tensor;
        }
        Tensor::new(BridgeTensor::Int(Dispatch::int_cast(
            tensor.primitive.into(),
            dtype,
        )))
    }
}

impl<const D: usize> Cast<D, Int> for FloatDType {
    type OutputKind = Float;

    fn cast(tensor: Tensor<D, Int>, dtype: Self) -> Tensor<D, Float> {
        Tensor::new(BridgeTensor::Float(Dispatch::int_into_float(
            tensor.primitive.into(),
            dtype,
        )))
    }
}

/// Backward-compatible impl: only int `DType` variants are accepted.
///
/// # Panics
///
/// Panics if `dtype` is not an int variant (e.g., `DType::F32`).
/// Use [`FloatDType`] directly for cross-kind casting to float.
impl<const D: usize> Cast<D, Int> for DType {
    type OutputKind = Int;

    fn cast(tensor: Tensor<D, Int>, dtype: Self) -> Tensor<D, Int> {
        let int_dtype: IntDType = dtype.into();
        <IntDType as Cast<D, Int>>::cast(tensor, int_dtype)
    }
}

// --- Bool input impls ---

impl<const D: usize> Cast<D, Bool> for IntDType {
    type OutputKind = Int;

    fn cast(tensor: Tensor<D, Bool>, dtype: Self) -> Tensor<D, Int> {
        Tensor::new(BridgeTensor::Bool(Dispatch::bool_into_int(
            tensor.primitive.into(),
            dtype,
        )))
    }
}

impl<const D: usize> Cast<D, Bool> for FloatDType {
    type OutputKind = Float;

    fn cast(tensor: Tensor<D, Bool>, dtype: Self) -> Tensor<D, Float> {
        Tensor::new(BridgeTensor::Float(Dispatch::bool_into_float(
            tensor.primitive.into(),
            dtype,
        )))
    }
}
