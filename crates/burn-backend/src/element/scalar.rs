use burn_std::{DType, bf16, f16};
use num_traits::ToPrimitive;

use crate::{Element, ElementConversion};

/// A scalar element.
#[derive(Clone, Copy, Debug)]
#[allow(missing_docs)]
pub enum Scalar {
    Float(f64),
    Int(i64),
    UInt(u64),
    Bool(bool),
}

impl Scalar {
    /// Creates a scalar with the specified data type.
    ///
    /// # Note
    /// [`QFloat`](DType::QFloat) scalars are represented as float for element-wise operations.
    pub fn new<E: ElementConversion>(value: E, dtype: &DType) -> Self {
        if dtype.is_float() | matches!(dtype, &DType::QFloat(_)) {
            Self::Float(value.elem())
        } else if dtype.is_int() {
            Self::Int(value.elem())
        } else if dtype.is_uint() {
            Self::UInt(value.elem())
        } else if dtype.is_bool() {
            Self::Bool(value.elem())
        } else {
            unimplemented!("Scalar not supported for {dtype:?}")
        }
    }

    /// Converts and returns the converted element.
    pub fn elem<E: Element>(self) -> E {
        match self {
            Self::Float(x) => x.elem(),
            Self::Int(x) => x.elem(),
            Self::UInt(x) => x.elem(),
            Self::Bool(x) => x.elem(),
        }
    }

    /// Returns the exact integer value, if valid.
    pub fn try_as_integer(&self) -> Option<Self> {
        match self {
            Scalar::Float(x) => (x.floor() == *x).then(|| Self::Int(x.to_i64().unwrap())),
            Scalar::Int(_) | Scalar::UInt(_) => Some(*self),
            Scalar::Bool(x) => Some(Scalar::Int(*x as i64)),
        }
    }
}

macro_rules! impl_from_scalar {
    ($($ty:ty => $variant:ident),+ $(,)?) => {
        $(
            impl From<$ty> for Scalar {
                fn from(value: $ty) -> Self {
                    Scalar::$variant(value.elem())
                }
            }
        )+
    };
}

impl_from_scalar! {
    f64  => Float, f32  => Float, f16  => Float, bf16 => Float,
    i64  => Int, i32  => Int, i16  => Int, i8 => Int,
    u64  => UInt, u32  => UInt, u16  => UInt, u8 => UInt, bool => Bool,
}

// CubeCL requirement
impl ToPrimitive for Scalar {
    fn to_i64(&self) -> Option<i64> {
        match self {
            Scalar::Float(x) => x.to_i64(),
            Scalar::UInt(x) => x.to_i64(),
            Scalar::Int(x) => Some(*x),
            Scalar::Bool(x) => Some(*x as i64),
        }
    }

    fn to_u64(&self) -> Option<u64> {
        match self {
            Scalar::Float(x) => x.to_u64(),
            Scalar::UInt(x) => Some(*x),
            Scalar::Int(x) => x.to_u64(),
            Scalar::Bool(x) => Some(*x as u64),
        }
    }

    fn to_f64(&self) -> Option<f64> {
        match self {
            Scalar::Float(x) => Some(*x),
            Scalar::UInt(x) => x.to_f64(),
            Scalar::Int(x) => x.to_f64(),
            Scalar::Bool(x) => (*x as u8).to_f64(),
        }
    }
}
