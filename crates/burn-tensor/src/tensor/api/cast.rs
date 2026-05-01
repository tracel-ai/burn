use burn_backend::ops::{BoolTensorOps, FloatTensorOps, IntTensorOps};
use burn_backend::tensor::{Bool, BoolTensor, Float, Int, IntTensor, TensorKind};
use burn_backend::{DType, FloatDType, IntDType, TensorMetadata, TensorPrimitive};
use burn_dispatch::Dispatch;

use crate::kind::Basic;

/// Trait for types that represent a valid cast target from a tensor of kind `K`.
///
/// The generic parameter `K` is the *input* tensor kind ([`Float`], [`Int`], or [`Bool`]).
/// Implementors declare the output kind and provide the actual cast logic.
pub trait Cast<K: Basic> {
    /// The output tensor kind after casting.
    type OutputKind: Basic;

    /// Cast a tensor primitive to the target dtype.
    fn cast(
        primitive: <K as TensorKind<Dispatch>>::Primitive,
        dtype: Self,
    ) -> <Self::OutputKind as TensorKind<Dispatch>>::Primitive;
}

// --- Float input impls ---

impl Cast<Float> for FloatDType {
    type OutputKind = Float;

    fn cast(primitive: TensorPrimitive<Dispatch>, dtype: Self) -> TensorPrimitive<Dispatch> {
        if let TensorPrimitive::Float(ref tensor) = primitive {
            let current: FloatDType = tensor.dtype().into();
            if current == dtype {
                return primitive;
            }
        }
        TensorPrimitive::Float(Dispatch::float_cast(primitive.tensor(), dtype))
    }
}

impl Cast<Float> for IntDType {
    type OutputKind = Int;

    fn cast(primitive: TensorPrimitive<Dispatch>, dtype: Self) -> IntTensor<Dispatch> {
        Dispatch::float_into_int(primitive.tensor(), dtype)
    }
}

/// Backward-compatible impl: only float `DType` variants are accepted.
///
/// # Panics
///
/// Panics if `dtype` is not a float variant (e.g., `DType::I32`).
/// Use [`IntDType`] directly for cross-kind casting to int.
impl Cast<Float> for DType {
    type OutputKind = Float;

    fn cast(primitive: TensorPrimitive<Dispatch>, dtype: Self) -> TensorPrimitive<Dispatch> {
        let float_dtype: FloatDType = dtype.into();
        <FloatDType as Cast<Float>>::cast(primitive, float_dtype)
    }
}

// --- Int input impls ---

impl Cast<Int> for IntDType {
    type OutputKind = Int;

    fn cast(primitive: IntTensor<Dispatch>, dtype: Self) -> IntTensor<Dispatch> {
        let current: IntDType = primitive.dtype().into();
        if current == dtype {
            return primitive;
        }
        Dispatch::int_cast(primitive, dtype)
    }
}

impl Cast<Int> for FloatDType {
    type OutputKind = Float;

    fn cast(primitive: IntTensor<Dispatch>, dtype: Self) -> TensorPrimitive<Dispatch> {
        TensorPrimitive::Float(Dispatch::int_into_float(primitive, dtype))
    }
}

/// Backward-compatible impl: only int `DType` variants are accepted.
///
/// # Panics
///
/// Panics if `dtype` is not an int variant (e.g., `DType::F32`).
/// Use [`FloatDType`] directly for cross-kind casting to float.
impl Cast<Int> for DType {
    type OutputKind = Int;

    fn cast(primitive: IntTensor<Dispatch>, dtype: Self) -> IntTensor<Dispatch> {
        let int_dtype: IntDType = dtype.into();
        <IntDType as Cast<Int>>::cast(primitive, int_dtype)
    }
}

// --- Bool input impls ---

impl Cast<Bool> for IntDType {
    type OutputKind = Int;

    fn cast(primitive: BoolTensor<Dispatch>, dtype: Self) -> IntTensor<Dispatch> {
        Dispatch::bool_into_int(primitive, dtype)
    }
}

impl Cast<Bool> for FloatDType {
    type OutputKind = Float;

    fn cast(primitive: BoolTensor<Dispatch>, dtype: Self) -> TensorPrimitive<Dispatch> {
        TensorPrimitive::Float(Dispatch::bool_into_float(primitive, dtype))
    }
}
