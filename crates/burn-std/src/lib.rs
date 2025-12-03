#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Burn Standard Library
//!
//! This library contains core types and utilities shared across Burn, including shapes, indexing,
//! and data types.

extern crate alloc;

/// Id module contains types for unique identifiers.
pub mod id;

/// Tensor utilities.
pub mod tensor;
pub use tensor::*;

/// Network utilities.
#[cfg(feature = "network")]
pub mod network;

// Re-exported types
pub use cubecl_common::bytes::*;
pub use cubecl_common::*;
pub use half::{bf16, f16};

#[cfg(feature = "cubecl")]
pub use cubecl::flex32;

#[cfg(feature = "cubecl")]
mod cube {
    use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, UIntKind};
    use cubecl_quant::scheme::QuantScheme;

    use crate::tensor::DType;
    use crate::tensor::quantization::{QuantStore, QuantValue};

    impl From<DType> for cubecl::ir::ElemType {
        fn from(dtype: DType) -> Self {
            match dtype {
                DType::F64 => ElemType::Float(FloatKind::F64),
                DType::F32 => ElemType::Float(FloatKind::F32),
                DType::Flex32 => ElemType::Float(FloatKind::Flex32),
                DType::F16 => ElemType::Float(FloatKind::F16),
                DType::BF16 => ElemType::Float(FloatKind::BF16),
                DType::I64 => ElemType::Int(IntKind::I64),
                DType::I32 => ElemType::Int(IntKind::I32),
                DType::I16 => ElemType::Int(IntKind::I16),
                DType::I8 => ElemType::Int(IntKind::I8),
                DType::U64 => ElemType::UInt(UIntKind::U64),
                DType::U32 => ElemType::UInt(UIntKind::U32),
                DType::U16 => ElemType::UInt(UIntKind::U16),
                DType::U8 => ElemType::UInt(UIntKind::U8),
                DType::Bool => ElemType::Bool,
                DType::QFloat(scheme) => match scheme.store {
                    QuantStore::Native => match scheme.value {
                        QuantValue::Q8F | QuantValue::Q8S => Self::Int(IntKind::I8),
                        QuantValue::E4M3 => Self::Float(FloatKind::E4M3),
                        QuantValue::E5M2 => Self::Float(FloatKind::E5M2),
                        QuantValue::Q4F
                        | QuantValue::Q4S
                        | QuantValue::Q2F
                        | QuantValue::Q2S
                        | QuantValue::E2M1 => {
                            panic!("Can't store native sub-byte values")
                        }
                    },
                    QuantStore::U32 => Self::UInt(UIntKind::U32),
                },
            }
        }
    }

    impl From<DType> for cubecl::ir::StorageType {
        fn from(dtype: DType) -> cubecl::ir::StorageType {
            match dtype {
                DType::QFloat(QuantScheme {
                    store: QuantStore::Native,
                    value: QuantValue::E2M1,
                    ..
                }) => StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2),
                _ => {
                    let elem: ElemType = dtype.into();
                    elem.into()
                }
            }
        }
    }
}
