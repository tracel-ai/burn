use burn_backend::tensor::{Bool, Float, Int, TensorKind};
use burn_backend::{Backend, DType, FloatDType, IntDType, TensorMetadata, TensorPrimitive};

/// Trait for types that represent a valid cast target from a tensor of kind `K`.
///
/// The generic parameter `K` is the *input* tensor kind ([`Float`], [`Int`], or [`Bool`]).
/// Implementors declare the output kind and provide the actual cast logic.
pub trait Cast<B: Backend, K: TensorKind<B>> {
    /// The output tensor kind after casting.
    type OutputKind: TensorKind<B>;

    /// Cast a tensor primitive to the target dtype.
    fn cast(primitive: K::Primitive, dtype: Self)
    -> <Self::OutputKind as TensorKind<B>>::Primitive;
}

// --- Float input impls ---

impl<B: Backend> Cast<B, Float> for FloatDType {
    type OutputKind = Float;

    fn cast(primitive: TensorPrimitive<B>, dtype: Self) -> TensorPrimitive<B> {
        if let TensorPrimitive::Float(ref tensor) = primitive {
            let current: FloatDType = tensor.dtype().into();
            if current == dtype {
                return primitive;
            }
        }
        TensorPrimitive::Float(B::float_cast(primitive.tensor(), dtype))
    }
}

impl<B: Backend> Cast<B, Float> for IntDType {
    type OutputKind = Int;

    fn cast(primitive: TensorPrimitive<B>, dtype: Self) -> B::IntTensorPrimitive {
        B::float_into_int(primitive.tensor(), dtype)
    }
}

/// Backward-compatible impl: only float `DType` variants are accepted.
///
/// # Panics
///
/// Panics if `dtype` is not a float variant (e.g., `DType::I32`).
/// Use [`IntDType`] directly for cross-kind casting to int.
impl<B: Backend> Cast<B, Float> for DType {
    type OutputKind = Float;

    fn cast(primitive: TensorPrimitive<B>, dtype: Self) -> TensorPrimitive<B> {
        let float_dtype: FloatDType = dtype.into();
        <FloatDType as Cast<B, Float>>::cast(primitive, float_dtype)
    }
}

// --- Int input impls ---

impl<B: Backend> Cast<B, Int> for IntDType {
    type OutputKind = Int;

    fn cast(primitive: B::IntTensorPrimitive, dtype: Self) -> B::IntTensorPrimitive {
        let current: IntDType = primitive.dtype().into();
        if current == dtype {
            return primitive;
        }
        B::int_cast(primitive, dtype)
    }
}

impl<B: Backend> Cast<B, Int> for FloatDType {
    type OutputKind = Float;

    fn cast(primitive: B::IntTensorPrimitive, dtype: Self) -> TensorPrimitive<B> {
        TensorPrimitive::Float(B::int_into_float(primitive, dtype))
    }
}

/// Backward-compatible impl: only int `DType` variants are accepted.
///
/// # Panics
///
/// Panics if `dtype` is not an int variant (e.g., `DType::F32`).
/// Use [`FloatDType`] directly for cross-kind casting to float.
impl<B: Backend> Cast<B, Int> for DType {
    type OutputKind = Int;

    fn cast(primitive: B::IntTensorPrimitive, dtype: Self) -> B::IntTensorPrimitive {
        let int_dtype: IntDType = dtype.into();
        <IntDType as Cast<B, Int>>::cast(primitive, int_dtype)
    }
}

// --- Bool input impls ---

impl<B: Backend> Cast<B, Bool> for IntDType {
    type OutputKind = Int;

    fn cast(primitive: B::BoolTensorPrimitive, dtype: Self) -> B::IntTensorPrimitive {
        B::bool_into_int(primitive, dtype)
    }
}

impl<B: Backend> Cast<B, Bool> for FloatDType {
    type OutputKind = Float;

    fn cast(primitive: B::BoolTensorPrimitive, dtype: Self) -> TensorPrimitive<B> {
        TensorPrimitive::Float(B::bool_into_float(primitive, dtype))
    }
}
