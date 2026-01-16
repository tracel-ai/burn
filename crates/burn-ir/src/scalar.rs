use burn_backend::{DType, Scalar, bf16, f16};
use burn_backend::{Element, ElementConversion};
use core::hash::Hash;
use serde::{Deserialize, Serialize};

/// A scalar representation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum ScalarIr {
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

impl Hash for ScalarIr {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        match self {
            ScalarIr::F64(x) => x.to_bits().hash(state),
            ScalarIr::F32(x) => x.to_bits().hash(state),
            ScalarIr::F16(x) => x.to_bits().hash(state),
            ScalarIr::BF16(x) => x.to_bits().hash(state),
            ScalarIr::I64(x) => x.hash(state),
            ScalarIr::I32(x) => x.hash(state),
            ScalarIr::I16(x) => x.hash(state),
            ScalarIr::I8(x) => x.hash(state),
            ScalarIr::U64(x) => x.hash(state),
            ScalarIr::U32(x) => x.hash(state),
            ScalarIr::U16(x) => x.hash(state),
            ScalarIr::U8(x) => x.hash(state),
            ScalarIr::Bool(x) => x.hash(state),
        }
    }
}

impl ScalarIr {
    /// Converts and returns the converted element.
    pub fn elem<E: Element>(self) -> E {
        match self {
            ScalarIr::F64(x) => x.elem(),
            ScalarIr::F32(x) => x.elem(),
            ScalarIr::F16(x) => x.elem(),
            ScalarIr::BF16(x) => x.elem(),
            ScalarIr::I64(x) => x.elem(),
            ScalarIr::I32(x) => x.elem(),
            ScalarIr::I16(x) => x.elem(),
            ScalarIr::I8(x) => x.elem(),
            ScalarIr::U64(x) => x.elem(),
            ScalarIr::U32(x) => x.elem(),
            ScalarIr::U16(x) => x.elem(),
            ScalarIr::U8(x) => x.elem(),
            ScalarIr::Bool(x) => x.elem(),
        }
    }

    /// Creates a scalar with the specified data type.
    pub fn with_dtype<E: Element>(elem: E, dtype: &DType) -> Self {
        match dtype {
            DType::F64 => ScalarIr::F64(elem.elem()),
            DType::F32 => ScalarIr::F32(elem.elem()),
            DType::Flex32 => ScalarIr::F32(elem.elem()),
            DType::F16 => ScalarIr::F16(elem.elem()),
            DType::BF16 => ScalarIr::BF16(elem.elem()),
            DType::I64 => ScalarIr::I64(elem.elem()),
            DType::I32 => ScalarIr::I32(elem.elem()),
            DType::I16 => ScalarIr::I16(elem.elem()),
            DType::I8 => ScalarIr::I8(elem.elem()),
            DType::U64 => ScalarIr::U64(elem.elem()),
            DType::U32 => ScalarIr::U32(elem.elem()),
            DType::U16 => ScalarIr::U16(elem.elem()),
            DType::U8 => ScalarIr::U8(elem.elem()),
            DType::Bool => ScalarIr::Bool(elem.elem()),
            DType::QFloat(_) => unimplemented!(),
        }
    }
}

// The enums are similar, but both types have different roles:
// - `Scalar`: runtime literal value
// - `ScalarIr`: serializable literal representation (used for IR)
impl From<Scalar> for ScalarIr {
    fn from(value: Scalar) -> Self {
        match value {
            Scalar::F64(x) => Self::F64(x),
            Scalar::F32(x) => Self::F32(x),
            Scalar::F16(x) => Self::F16(x),
            Scalar::BF16(x) => Self::BF16(x),
            Scalar::I64(x) => Self::I64(x),
            Scalar::I32(x) => Self::I32(x),
            Scalar::I16(x) => Self::I16(x),
            Scalar::I8(x) => Self::I8(x),
            Scalar::U64(x) => Self::U64(x),
            Scalar::U32(x) => Self::U32(x),
            Scalar::U16(x) => Self::U16(x),
            Scalar::U8(x) => Self::U8(x),
            Scalar::Bool(x) => Self::Bool(x),
        }
    }
}

impl From<ScalarIr> for Scalar {
    fn from(value: ScalarIr) -> Self {
        match value {
            ScalarIr::F64(x) => Self::F64(x),
            ScalarIr::F32(x) => Self::F32(x),
            ScalarIr::F16(x) => Self::F16(x),
            ScalarIr::BF16(x) => Self::BF16(x),
            ScalarIr::I64(x) => Self::I64(x),
            ScalarIr::I32(x) => Self::I32(x),
            ScalarIr::I16(x) => Self::I16(x),
            ScalarIr::I8(x) => Self::I8(x),
            ScalarIr::U64(x) => Self::U64(x),
            ScalarIr::U32(x) => Self::U32(x),
            ScalarIr::U16(x) => Self::U16(x),
            ScalarIr::U8(x) => Self::U8(x),
            ScalarIr::Bool(x) => Self::Bool(x),
        }
    }
}
