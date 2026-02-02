use burn_backend::{DType, Scalar};
use burn_backend::{Element, ElementConversion};
use core::hash::Hash;
use serde::{Deserialize, Serialize};

/// A scalar representation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum ScalarIr {
    Float(f64),
    Int(i64),
    UInt(u64),
    Bool(bool),
}

impl Hash for ScalarIr {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        match self {
            ScalarIr::Float(x) => x.to_bits().hash(state),
            ScalarIr::Int(x) => x.hash(state),
            ScalarIr::UInt(x) => x.hash(state),
            ScalarIr::Bool(x) => x.hash(state),
        }
    }
}

impl ScalarIr {
    /// Creates a scalar with the specified data type.
    pub fn new<E: ElementConversion>(value: E, dtype: &DType) -> Self {
        if dtype.is_float() {
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
            ScalarIr::Float(x) => x.elem(),
            ScalarIr::Int(x) => x.elem(),
            ScalarIr::UInt(x) => x.elem(),
            ScalarIr::Bool(x) => x.elem(),
        }
    }
}

// The enums are similar, but both types have different roles:
// - `Scalar`: runtime literal value
// - `ScalarIr`: serializable literal representation (used for IR)
impl From<Scalar> for ScalarIr {
    fn from(value: Scalar) -> Self {
        match value {
            Scalar::Float(x) => Self::Float(x),
            Scalar::Int(x) => Self::Int(x),
            Scalar::UInt(x) => Self::UInt(x),
            Scalar::Bool(x) => Self::Bool(x),
        }
    }
}

impl From<ScalarIr> for Scalar {
    fn from(value: ScalarIr) -> Self {
        match value {
            ScalarIr::Float(x) => Self::Float(x),
            ScalarIr::Int(x) => Self::Int(x),
            ScalarIr::UInt(x) => Self::UInt(x),
            ScalarIr::Bool(x) => Self::Bool(x),
            DType::Complex64 => todo!(),
            DType::Complex32 => todo!(),
        }
    }
}
