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
        if tensor.primitive.is_float() {
            let current: FloatDType = tensor.dtype().into();
            if current == dtype {
                return tensor;
            }
            Tensor::new(float_cast_impl(tensor.primitive, dtype))
        } else {
            panic!("Should be Float primitive kind");
        }
    }
}

impl<const D: usize> Cast<D, Float> for IntDType {
    type OutputKind = Int;

    fn cast(tensor: Tensor<D, Float>, dtype: Self) -> Tensor<D, Int> {
        Tensor::new(float_to_int_impl(tensor.primitive, dtype))
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
        Tensor::new(int_cast_impl(tensor.primitive, dtype))
    }
}

impl<const D: usize> Cast<D, Int> for FloatDType {
    type OutputKind = Float;

    fn cast(tensor: Tensor<D, Int>, dtype: Self) -> Tensor<D, Float> {
        Tensor::new(int_to_float_impl(tensor.primitive, dtype))
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
        Tensor::new(bool_cast_to_int_impl(tensor.primitive, dtype))
    }
}

impl<const D: usize> Cast<D, Bool> for FloatDType {
    type OutputKind = Float;

    fn cast(tensor: Tensor<D, Bool>, dtype: Self) -> Tensor<D, Float> {
        Tensor::new(bool_cast_to_float_impl(tensor.primitive, dtype))
    }
}

// =========================================================================
// Non-generic implementation helpers (outlined from the generic API).
// See the crate-level docs for the rationale behind this pattern.
// =========================================================================

fn float_cast_impl(p: BridgeTensor, dtype: FloatDType) -> BridgeTensor {
    BridgeTensor::float(Dispatch::float_cast(p.into_float(), dtype))
}

fn float_to_int_impl(p: BridgeTensor, dtype: IntDType) -> BridgeTensor {
    BridgeTensor::int(Dispatch::float_into_int(p.into_float(), dtype))
}

fn int_cast_impl(p: BridgeTensor, dtype: IntDType) -> BridgeTensor {
    BridgeTensor::int(Dispatch::int_cast(p.into(), dtype))
}

fn int_to_float_impl(p: BridgeTensor, dtype: FloatDType) -> BridgeTensor {
    BridgeTensor::float(Dispatch::int_into_float(p.into(), dtype))
}

fn bool_cast_to_int_impl(p: BridgeTensor, dtype: IntDType) -> BridgeTensor {
    BridgeTensor::bool(Dispatch::bool_into_int(p.into(), dtype))
}

fn bool_cast_to_float_impl(p: BridgeTensor, dtype: FloatDType) -> BridgeTensor {
    BridgeTensor::float(Dispatch::bool_into_float(p.into(), dtype))
}
