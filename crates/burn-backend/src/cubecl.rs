//! Conversion helpers between burn's [`DType`] and cubecl's
//! `ElemType` / `StorageType`.
//!
//! These are exposed as plain named functions (rather than `From`/`Into`
//! trait impls in `burn-std`) so that `burn-std` doesn't need to depend on
//! `cubecl` and the cubecl type tree doesn't leak into the shared standard
//! library. Backend implementations that want these conversions depend on
//! `burn-backend` (with the `cubecl` feature) and call these functions
//! explicitly.

use burn_std::{BoolStore, DType, QuantScheme, QuantStore, QuantValue};
use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, UIntKind};

/// Convert a cubecl [`ElemType`] into the corresponding burn [`DType`].
///
/// Panics if the cubecl type has no direct burn equivalent (e.g. `TF32`).
pub fn elem_type_to_dtype(value: ElemType) -> DType {
    match value {
        ElemType::Float(float_kind) => match float_kind {
            FloatKind::F16 => DType::F16,
            FloatKind::BF16 => DType::BF16,
            FloatKind::Flex32 => DType::Flex32,
            FloatKind::F32 => DType::F32,
            FloatKind::F64 => DType::F64,
            FloatKind::TF32 => panic!("Not a valid DType for tensors."),
            FloatKind::E2M1
            | FloatKind::E2M3
            | FloatKind::E3M2
            | FloatKind::E4M3
            | FloatKind::E5M2
            | FloatKind::UE8M0 => {
                unimplemented!("Not yet supported, will be used for quantization")
            }
        },
        ElemType::Int(int_kind) => match int_kind {
            IntKind::I8 => DType::I8,
            IntKind::I16 => DType::I16,
            IntKind::I32 => DType::I32,
            IntKind::I64 => DType::I64,
        },
        ElemType::UInt(uint_kind) => match uint_kind {
            UIntKind::U8 => DType::U8,
            UIntKind::U16 => DType::U16,
            UIntKind::U32 => DType::U32,
            UIntKind::U64 => DType::U64,
        },
        _ => panic!("Not a valid DType for tensors."),
    }
}

/// Convert a burn [`DType`] into the corresponding cubecl [`ElemType`].
///
/// Sub-byte quantization variants whose storage is `PackedNative` may panic
/// (see inline `panic!`s) — those configurations are not representable as a
/// single `ElemType`; use [`dtype_to_storage_type`] instead.
pub fn dtype_to_elem_type(dtype: DType) -> ElemType {
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
        DType::Bool(store) => match store {
            BoolStore::Native => ElemType::Bool,
            BoolStore::U8 => ElemType::UInt(UIntKind::U8),
            BoolStore::U32 => ElemType::UInt(UIntKind::U32),
        },
        DType::QFloat(scheme) => match scheme.store {
            QuantStore::Native => match scheme.value {
                QuantValue::Q8F | QuantValue::Q8S => ElemType::Int(IntKind::I8),
                QuantValue::E4M3 => ElemType::Float(FloatKind::E4M3),
                QuantValue::E5M2 => ElemType::Float(FloatKind::E5M2),
                QuantValue::Q4F
                | QuantValue::Q4S
                | QuantValue::Q2F
                | QuantValue::Q2S
                | QuantValue::E2M1 => {
                    panic!("Can't store native sub-byte values")
                }
            },
            QuantStore::PackedU32(_) => ElemType::UInt(UIntKind::U32),
            QuantStore::PackedNative(_) => match scheme.value {
                QuantValue::E2M1 => panic!("Can't store native sub-byte values"),
                other => panic!("{other:?} doesn't support native packing"),
            },
        },
    }
}

/// Convert a burn [`DType`] into the corresponding cubecl [`StorageType`].
///
/// Handles sub-byte packed quantization configurations that cannot be expressed
/// as a plain [`ElemType`] by emitting a `StorageType::Packed(...)`.
pub fn dtype_to_storage_type(dtype: DType) -> StorageType {
    match dtype {
        DType::QFloat(QuantScheme {
            store: QuantStore::PackedNative(_),
            value: QuantValue::E2M1,
            ..
        }) => StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2),
        _ => dtype_to_elem_type(dtype).into(),
    }
}
