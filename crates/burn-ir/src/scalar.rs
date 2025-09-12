use burn_tensor::{DType, Element, ElementConversion, bf16, f16};
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
