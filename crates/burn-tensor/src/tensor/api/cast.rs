use burn_backend::tensor::{Float, Int, TensorKind};
use burn_backend::{Backend, DType, FloatDType, IntDType, TensorMetadata, TensorPrimitive};

/// Trait for types that represent a valid cast target from a float tensor.
///
/// Implemented for [`FloatDType`] (within-kind), [`IntDType`] (cross-kind),
/// and [`DType`] (backward-compatible within-kind; panics if given a non-float variant).
pub trait CastFromFloat<B: Backend> {
    /// The output tensor kind after casting.
    type OutputKind: TensorKind<B>;

    /// Cast a float tensor primitive to the target dtype.
    fn cast_from_float(
        primitive: TensorPrimitive<B>,
        dtype: Self,
    ) -> <Self::OutputKind as TensorKind<B>>::Primitive;
}

/// Trait for types that represent a valid cast target from an int tensor.
///
/// Implemented for [`IntDType`] (within-kind), [`FloatDType`] (cross-kind),
/// and [`DType`] (backward-compatible within-kind; panics if given a non-int variant).
pub trait CastFromInt<B: Backend> {
    /// The output tensor kind after casting.
    type OutputKind: TensorKind<B>;

    /// Cast an int tensor primitive to the target dtype.
    fn cast_from_int(
        primitive: B::IntTensorPrimitive,
        dtype: Self,
    ) -> <Self::OutputKind as TensorKind<B>>::Primitive;
}

/// Trait for types that represent a valid cast target from a bool tensor.
///
/// Implemented for [`IntDType`] and [`FloatDType`].
pub trait CastFromBool<B: Backend> {
    /// The output tensor kind after casting.
    type OutputKind: TensorKind<B>;

    /// Cast a bool tensor primitive to the target dtype.
    fn cast_from_bool(
        primitive: B::BoolTensorPrimitive,
        dtype: Self,
    ) -> <Self::OutputKind as TensorKind<B>>::Primitive;
}

impl<B: Backend> CastFromFloat<B> for FloatDType {
    type OutputKind = Float;

    fn cast_from_float(primitive: TensorPrimitive<B>, dtype: Self) -> TensorPrimitive<B> {
        if let TensorPrimitive::Float(ref tensor) = primitive {
            let current: FloatDType = tensor.dtype().into();
            if current == dtype {
                return primitive;
            }
        }
        TensorPrimitive::Float(B::float_cast(primitive.tensor(), dtype))
    }
}

impl<B: Backend> CastFromFloat<B> for IntDType {
    type OutputKind = Int;

    fn cast_from_float(primitive: TensorPrimitive<B>, dtype: Self) -> B::IntTensorPrimitive {
        B::float_into_int(primitive.tensor(), dtype)
    }
}

/// Backward-compatible impl: only float `DType` variants are accepted.
///
/// # Panics
///
/// Panics if `dtype` is not a float variant (e.g., `DType::I32`).
/// Use [`IntDType`] directly for cross-kind casting to int.
impl<B: Backend> CastFromFloat<B> for DType {
    type OutputKind = Float;

    fn cast_from_float(primitive: TensorPrimitive<B>, dtype: Self) -> TensorPrimitive<B> {
        let float_dtype: FloatDType = dtype.into();
        <FloatDType as CastFromFloat<B>>::cast_from_float(primitive, float_dtype)
    }
}

impl<B: Backend> CastFromInt<B> for IntDType {
    type OutputKind = Int;

    fn cast_from_int(primitive: B::IntTensorPrimitive, dtype: Self) -> B::IntTensorPrimitive {
        let current: IntDType = primitive.dtype().into();
        if current == dtype {
            return primitive;
        }
        B::int_cast(primitive, dtype)
    }
}

impl<B: Backend> CastFromInt<B> for FloatDType {
    type OutputKind = Float;

    fn cast_from_int(primitive: B::IntTensorPrimitive, dtype: Self) -> TensorPrimitive<B> {
        TensorPrimitive::Float(B::int_into_float(primitive, dtype))
    }
}

/// Backward-compatible impl: only int `DType` variants are accepted.
///
/// # Panics
///
/// Panics if `dtype` is not an int variant (e.g., `DType::F32`).
/// Use [`FloatDType`] directly for cross-kind casting to float.
impl<B: Backend> CastFromInt<B> for DType {
    type OutputKind = Int;

    fn cast_from_int(primitive: B::IntTensorPrimitive, dtype: Self) -> B::IntTensorPrimitive {
        let int_dtype: IntDType = dtype.into();
        <IntDType as CastFromInt<B>>::cast_from_int(primitive, int_dtype)
    }
}

impl<B: Backend> CastFromBool<B> for IntDType {
    type OutputKind = Int;

    fn cast_from_bool(primitive: B::BoolTensorPrimitive, dtype: Self) -> B::IntTensorPrimitive {
        B::bool_into_int(primitive, dtype)
    }
}

impl<B: Backend> CastFromBool<B> for FloatDType {
    type OutputKind = Float;

    fn cast_from_bool(primitive: B::BoolTensorPrimitive, dtype: Self) -> TensorPrimitive<B> {
        TensorPrimitive::Float(B::bool_into_float(primitive, dtype))
    }
}
