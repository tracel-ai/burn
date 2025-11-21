use serde::{Deserialize, Serialize};

use crate::quantization::{QuantScheme, QuantStore, QuantValue};
use crate::{bf16, f16};

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F64,
    F32,
    Flex32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
    QFloat(QuantScheme),
}

#[cfg(feature = "cubecl")]
impl From<cubecl::ir::ElemType> for DType {
    fn from(value: cubecl::ir::ElemType) -> Self {
        match value {
            cubecl::ir::ElemType::Float(float_kind) => match float_kind {
                cubecl::ir::FloatKind::F16 => DType::F16,
                cubecl::ir::FloatKind::BF16 => DType::BF16,
                cubecl::ir::FloatKind::Flex32 => DType::Flex32,
                cubecl::ir::FloatKind::F32 => DType::F32,
                cubecl::ir::FloatKind::F64 => DType::F64,
                cubecl::ir::FloatKind::TF32 => panic!("Not a valid DType for tensors."),
                cubecl::ir::FloatKind::E2M1
                | cubecl::ir::FloatKind::E2M3
                | cubecl::ir::FloatKind::E3M2
                | cubecl::ir::FloatKind::E4M3
                | cubecl::ir::FloatKind::E5M2
                | cubecl::ir::FloatKind::UE8M0 => {
                    unimplemented!("Not yet supported, will be used for quantization")
                }
            },
            cubecl::ir::ElemType::Int(int_kind) => match int_kind {
                cubecl::ir::IntKind::I8 => DType::I8,
                cubecl::ir::IntKind::I16 => DType::I16,
                cubecl::ir::IntKind::I32 => DType::I32,
                cubecl::ir::IntKind::I64 => DType::I64,
            },
            cubecl::ir::ElemType::UInt(uint_kind) => match uint_kind {
                cubecl::ir::UIntKind::U8 => DType::U8,
                cubecl::ir::UIntKind::U16 => DType::U16,
                cubecl::ir::UIntKind::U32 => DType::U32,
                cubecl::ir::UIntKind::U64 => DType::U64,
            },
            _ => panic!("Not a valid DType for tensors."),
        }
    }
}

impl DType {
    /// Returns the size of a type in bytes.
    pub const fn size(&self) -> usize {
        match self {
            DType::F64 => core::mem::size_of::<f64>(),
            DType::F32 => core::mem::size_of::<f32>(),
            DType::Flex32 => core::mem::size_of::<f32>(),
            DType::F16 => core::mem::size_of::<f16>(),
            DType::BF16 => core::mem::size_of::<bf16>(),
            DType::I64 => core::mem::size_of::<i64>(),
            DType::I32 => core::mem::size_of::<i32>(),
            DType::I16 => core::mem::size_of::<i16>(),
            DType::I8 => core::mem::size_of::<i8>(),
            DType::U64 => core::mem::size_of::<u64>(),
            DType::U32 => core::mem::size_of::<u32>(),
            DType::U16 => core::mem::size_of::<u16>(),
            DType::U8 => core::mem::size_of::<u8>(),
            DType::Bool => core::mem::size_of::<bool>(),
            DType::QFloat(scheme) => match scheme.store {
                QuantStore::Native => match scheme.value {
                    QuantValue::Q8F | QuantValue::Q8S => core::mem::size_of::<i8>(),
                    // e2m1 native is automatically packed by the kernels, so the actual storage is
                    // 8 bits wide.
                    QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1 => {
                        core::mem::size_of::<u8>()
                    }
                    QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S => {
                        // Sub-byte values have fractional size
                        0
                    }
                },
                QuantStore::U32 => core::mem::size_of::<u32>(),
            },
        }
    }
    /// Returns true if the data type is a floating point type.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DType::F64 | DType::F32 | DType::Flex32 | DType::F16 | DType::BF16
        )
    }
    /// Returns true if the data type is a signed integer type.
    pub fn is_int(&self) -> bool {
        matches!(self, DType::I64 | DType::I32 | DType::I16 | DType::I8)
    }
    /// Returns true if the data type is an unsigned integer type.
    pub fn is_uint(&self) -> bool {
        matches!(self, DType::U64 | DType::U32 | DType::U16 | DType::U8)
    }

    /// Returns true if the data type is a boolean type
    pub fn is_bool(&self) -> bool {
        matches!(self, DType::Bool)
    }

    /// Returns the data type name.
    pub fn name(&self) -> &'static str {
        match self {
            DType::F64 => "f64",
            DType::F32 => "f32",
            DType::Flex32 => "flex32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I64 => "i64",
            DType::I32 => "i32",
            DType::I16 => "i16",
            DType::I8 => "i8",
            DType::U64 => "u64",
            DType::U32 => "u32",
            DType::U16 => "u16",
            DType::U8 => "u8",
            DType::Bool => "bool",
            DType::QFloat(_) => "qfloat",
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FloatDType {
    F64,
    F32,
    Flex32,
    F16,
    BF16,
}

impl From<DType> for FloatDType {
    fn from(value: DType) -> Self {
        match value {
            DType::F64 => FloatDType::F64,
            DType::F32 => FloatDType::F32,
            DType::Flex32 => FloatDType::Flex32,
            DType::F16 => FloatDType::F16,
            DType::BF16 => FloatDType::BF16,
            _ => panic!("Expected float data type, got {value:?}"),
        }
    }
}

impl From<FloatDType> for DType {
    fn from(value: FloatDType) -> Self {
        match value {
            FloatDType::F64 => DType::F64,
            FloatDType::F32 => DType::F32,
            FloatDType::Flex32 => DType::Flex32,
            FloatDType::F16 => DType::F16,
            FloatDType::BF16 => DType::BF16,
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IntDType {
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
}

impl From<DType> for IntDType {
    fn from(value: DType) -> Self {
        match value {
            DType::I64 => IntDType::I64,
            DType::I32 => IntDType::I32,
            DType::I16 => IntDType::I16,
            DType::I8 => IntDType::I8,
            DType::U64 => IntDType::U64,
            DType::U32 => IntDType::U32,
            DType::U16 => IntDType::U16,
            DType::U8 => IntDType::U8,
            _ => panic!("Expected int data type, got {value:?}"),
        }
    }
}

impl From<IntDType> for DType {
    fn from(value: IntDType) -> Self {
        match value {
            IntDType::I64 => DType::I64,
            IntDType::I32 => DType::I32,
            IntDType::I16 => DType::I16,
            IntDType::I8 => DType::I8,
            IntDType::U64 => DType::U64,
            IntDType::U32 => DType::U32,
            IntDType::U16 => DType::U16,
            IntDType::U8 => DType::U8,
        }
    }
}
