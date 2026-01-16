use burn_std::{DType, bf16, f16};
use num_traits::ToPrimitive;

use crate::{Element, ElementConversion};

/// A scalar element.
#[derive(Clone, Copy, Debug)]
#[allow(missing_docs)]
pub enum Scalar {
    F64(f64),
    F32(f32),
    F16(f16),
    BF16(bf16),
    I64(i64),
    I32(i32),
    I16(i16),
    I8(i8),
    U64(u64),
    U32(u32),
    U16(u16),
    U8(u8),
    Bool(bool),
}

impl Scalar {
    /// Converts and returns the converted element.
    pub fn elem<E: Element>(self) -> E {
        match self {
            Self::F64(x) => x.elem(),
            Self::F32(x) => x.elem(),
            Self::F16(x) => x.elem(),
            Self::BF16(x) => x.elem(),
            Self::I64(x) => x.elem(),
            Self::I32(x) => x.elem(),
            Self::I16(x) => x.elem(),
            Self::I8(x) => x.elem(),
            Self::U64(x) => x.elem(),
            Self::U32(x) => x.elem(),
            Self::U16(x) => x.elem(),
            Self::U8(x) => x.elem(),
            Self::Bool(x) => x.elem(),
        }
    }

    /// Creates a scalar with the specified data type.
    pub fn with_dtype<E: ElementConversion>(elem: E, dtype: &DType) -> Self {
        match dtype {
            DType::F64 => Self::F64(elem.elem()),
            DType::F32 => Self::F32(elem.elem()),
            DType::Flex32 => Self::F32(elem.elem()),
            DType::F16 => Self::F16(elem.elem()),
            DType::BF16 => Self::BF16(elem.elem()),
            DType::I64 => Self::I64(elem.elem()),
            DType::I32 => Self::I32(elem.elem()),
            DType::I16 => Self::I16(elem.elem()),
            DType::I8 => Self::I8(elem.elem()),
            DType::U64 => Self::U64(elem.elem()),
            DType::U32 => Self::U32(elem.elem()),
            DType::U16 => Self::U16(elem.elem()),
            DType::U8 => Self::U8(elem.elem()),
            DType::Bool => Self::Bool(elem.elem()),
            DType::QFloat(_) => unimplemented!(),
        }
    }
}

macro_rules! impl_from_scalar {
    ($($ty:ty => $variant:ident),+ $(,)?) => {
        $(
            impl From<$ty> for Scalar {
                fn from(value: $ty) -> Self {
                    Scalar::$variant(value)
                }
            }
        )+
    };
}

impl_from_scalar! {
    f64  => F64, f32  => F32, f16  => F16, bf16 => BF16,
    i64  => I64, i32  => I32, i16  => I16, i8 => I8,
    u64  => U64, u32  => U32, u16  => U16, u8 => U8, bool => Bool,
}

// CubeCL requirement
impl ToPrimitive for Scalar {
    fn to_i64(&self) -> Option<i64> {
        match self {
            Scalar::F64(x) => x.to_i64(),
            Scalar::F32(x) => x.to_i64(),
            Scalar::F16(x) => x.to_i64(),
            Scalar::BF16(x) => x.to_i64(),
            Scalar::I64(x) => x.to_i64(),
            Scalar::I32(x) => x.to_i64(),
            Scalar::I16(x) => x.to_i64(),
            Scalar::I8(x) => x.to_i64(),
            Scalar::U64(x) => x.to_i64(),
            Scalar::U32(x) => x.to_i64(),
            Scalar::U16(x) => x.to_i64(),
            Scalar::U8(x) => x.to_i64(),
            Scalar::Bool(x) => (*x as u8).to_i64(),
        }
    }

    fn to_u64(&self) -> Option<u64> {
        match self {
            Scalar::F64(x) => x.to_u64(),
            Scalar::F32(x) => x.to_u64(),
            Scalar::F16(x) => x.to_u64(),
            Scalar::BF16(x) => x.to_u64(),
            Scalar::I64(x) => x.to_u64(),
            Scalar::I32(x) => x.to_u64(),
            Scalar::I16(x) => x.to_u64(),
            Scalar::I8(x) => x.to_u64(),
            Scalar::U64(x) => x.to_u64(),
            Scalar::U32(x) => x.to_u64(),
            Scalar::U16(x) => x.to_u64(),
            Scalar::U8(x) => x.to_u64(),
            Scalar::Bool(x) => (*x as u8).to_u64(),
        }
    }

    fn to_f32(&self) -> Option<f32> {
        match self {
            Scalar::F64(x) => x.to_f32(),
            Scalar::F32(x) => x.to_f32(),
            Scalar::F16(x) => x.to_f32(),
            Scalar::BF16(x) => x.to_f32(),
            Scalar::I64(x) => x.to_f32(),
            Scalar::I32(x) => x.to_f32(),
            Scalar::I16(x) => x.to_f32(),
            Scalar::I8(x) => x.to_f32(),
            Scalar::U64(x) => x.to_f32(),
            Scalar::U32(x) => x.to_f32(),
            Scalar::U16(x) => x.to_f32(),
            Scalar::U8(x) => x.to_f32(),
            Scalar::Bool(x) => (*x as u8).to_f32(),
        }
    }

    fn to_f64(&self) -> Option<f64> {
        match self {
            Scalar::F64(x) => x.to_f64(),
            Scalar::F32(x) => x.to_f64(),
            Scalar::F16(x) => x.to_f64(),
            Scalar::BF16(x) => x.to_f64(),
            Scalar::I64(x) => x.to_f64(),
            Scalar::I32(x) => x.to_f64(),
            Scalar::I16(x) => x.to_f64(),
            Scalar::I8(x) => x.to_f64(),
            Scalar::U64(x) => x.to_f64(),
            Scalar::U32(x) => x.to_f64(),
            Scalar::U16(x) => x.to_f64(),
            Scalar::U8(x) => x.to_f64(),
            Scalar::Bool(x) => (*x as u8).to_f64(),
        }
    }
}
