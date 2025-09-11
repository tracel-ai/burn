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

// Provide `elem.into()` and `scalar.elem()` for convenience
impl<E: Element> From<E> for ScalarIr {
    fn from(value: E) -> Self {
        match E::dtype() {
            DType::F64 => ScalarIr::F64(value.elem()),
            DType::F32 => ScalarIr::F32(value.elem()),
            DType::Flex32 => ScalarIr::F32(value.elem()),
            DType::F16 => ScalarIr::F16(value.elem()),
            DType::BF16 => ScalarIr::BF16(value.elem()),
            DType::I64 => ScalarIr::I64(value.elem()),
            DType::I32 => ScalarIr::I32(value.elem()),
            DType::I16 => ScalarIr::I16(value.elem()),
            DType::I8 => ScalarIr::I8(value.elem()),
            DType::U64 => ScalarIr::U64(value.elem()),
            DType::U32 => ScalarIr::U32(value.elem()),
            DType::U16 => ScalarIr::U16(value.elem()),
            DType::U8 => ScalarIr::U8(value.elem()),
            DType::Bool => ScalarIr::Bool(value.elem()),
            DType::QFloat(_) => unimplemented!(),
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

    /// Returns the scalar data type.
    pub fn dtype(&self) -> DType {
        match self {
            ScalarIr::F64(_) => DType::F64,
            ScalarIr::F32(_) => DType::F32,
            ScalarIr::F16(_) => DType::F16,
            ScalarIr::BF16(_) => DType::BF16,
            ScalarIr::I64(_) => DType::I64,
            ScalarIr::I32(_) => DType::I32,
            ScalarIr::I16(_) => DType::I16,
            ScalarIr::I8(_) => DType::I8,
            ScalarIr::U64(_) => DType::U64,
            ScalarIr::U32(_) => DType::U32,
            ScalarIr::U16(_) => DType::U16,
            ScalarIr::U8(_) => DType::U8,
            ScalarIr::Bool(_) => DType::Bool,
        }
    }

    /// Converts the scalar to the specified data type.
    pub fn convert(self, dtype: &DType) -> Self {
        match dtype {
            DType::F64 => ScalarIr::F64(self.elem()),
            DType::F32 => ScalarIr::F32(self.elem()),
            DType::Flex32 => ScalarIr::F32(self.elem()),
            DType::F16 => ScalarIr::F16(self.elem()),
            DType::BF16 => ScalarIr::BF16(self.elem()),
            DType::I64 => ScalarIr::I64(self.elem()),
            DType::I32 => ScalarIr::I32(self.elem()),
            DType::I16 => ScalarIr::I16(self.elem()),
            DType::I8 => ScalarIr::I8(self.elem()),
            DType::U64 => ScalarIr::U64(self.elem()),
            DType::U32 => ScalarIr::U32(self.elem()),
            DType::U16 => ScalarIr::U16(self.elem()),
            DType::U8 => ScalarIr::U8(self.elem()),
            DType::Bool => ScalarIr::Bool(self.elem()),
            DType::QFloat(_) => unimplemented!(),
        }
    }
}
