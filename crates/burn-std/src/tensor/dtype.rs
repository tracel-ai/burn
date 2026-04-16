//! Tensor data type.

use serde::{Deserialize, Serialize};

use crate::tensor::quantization::{QuantScheme, QuantStore, QuantValue};
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
    Bool(BoolStore),
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
            DType::Bool(store) => match store {
                BoolStore::Native => core::mem::size_of::<bool>(),
                BoolStore::U8 => core::mem::size_of::<u8>(),
                BoolStore::U32 => core::mem::size_of::<u32>(),
            },
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
                QuantStore::PackedU32(_) => core::mem::size_of::<u32>(),
                QuantStore::PackedNative(_) => match scheme.value {
                    QuantValue::E2M1 => core::mem::size_of::<u8>(),
                    _ => 0,
                },
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
        matches!(self, DType::Bool(_))
    }

    /// Returns float precision info if this is a float dtype, `None` otherwise.
    ///
    /// Analogous to `torch.finfo(dtype)` or `numpy.finfo(dtype)`.
    pub const fn finfo(&self) -> Option<FloatInfo> {
        match self {
            DType::F64 => Some(FloatDType::F64.finfo()),
            DType::F32 => Some(FloatDType::F32.finfo()),
            DType::Flex32 => Some(FloatDType::Flex32.finfo()),
            DType::F16 => Some(FloatDType::F16.finfo()),
            DType::BF16 => Some(FloatDType::BF16.finfo()),
            _ => None,
        }
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
            DType::Bool(store) => match store {
                BoolStore::Native => "bool",
                BoolStore::U8 => "bool(u8)",
                BoolStore::U32 => "bool(u32)",
            },
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

/// Numerical precision properties for a floating-point dtype.
///
/// Equivalent to NumPy's `finfo` / PyTorch's `torch.finfo`. All values are
/// widened to `f64` so they can be inspected without knowing the concrete
/// element type at compile time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatInfo {
    /// Machine epsilon: smallest value such that `1.0 + epsilon != 1.0`.
    pub epsilon: f64,
    /// Largest representable finite value.
    pub max: f64,
    /// Most negative representable finite value.
    pub min: f64,
    /// Smallest positive normal value.
    pub min_positive: f64,
}

impl FloatDType {
    /// Returns numerical precision properties for this float dtype.
    ///
    /// Analogous to `torch.finfo(dtype)` or `numpy.finfo(dtype)`.
    pub const fn finfo(self) -> FloatInfo {
        match self {
            FloatDType::F64 => FloatInfo {
                epsilon: f64::EPSILON,
                max: f64::MAX,
                min: f64::MIN,
                min_positive: f64::MIN_POSITIVE, // ~2.225e-308
            },
            FloatDType::F32 => FloatInfo {
                epsilon: f32::EPSILON as f64,
                max: f32::MAX as f64,
                min: f32::MIN as f64,
                min_positive: f32::MIN_POSITIVE as f64, // ~1.175e-38
            },
            // Flex32 stores as f32 but computes at reduced (f16-like) precision.
            // Use f16 precision limits so stability code stays safe.
            FloatDType::Flex32 => FloatInfo {
                epsilon: f16::EPSILON.to_f64_const(),
                max: f16::MAX.to_f64_const(),
                min: f16::MIN.to_f64_const(),
                min_positive: f16::MIN_POSITIVE.to_f64_const(), // ~6.104e-5
            },
            FloatDType::F16 => FloatInfo {
                epsilon: f16::EPSILON.to_f64_const(),
                max: f16::MAX.to_f64_const(),
                min: f16::MIN.to_f64_const(),
                min_positive: f16::MIN_POSITIVE.to_f64_const(), // ~6.104e-5
            },
            FloatDType::BF16 => FloatInfo {
                epsilon: bf16::EPSILON.to_f64_const(),
                max: bf16::MAX.to_f64_const(),
                min: bf16::MIN.to_f64_const(),
                min_positive: bf16::MIN_POSITIVE.to_f64_const(), // ~1.175e-38
            },
        }
    }
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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// Data type used to store boolean values.
pub enum BoolStore {
    /// Stored as native boolean type (e.g. `bool`).
    Native,
    /// Stored as 8-bit unsigned integer.
    U8,
    /// Stored as 32-bit unsigned integer.
    U32,
}

/// Boolean dtype.
///
/// This is currently an alias to [`BoolStore`], since it only varies by the storage representation.
pub type BoolDType = BoolStore;

#[allow(deprecated)]
impl From<DType> for BoolDType {
    fn from(value: DType) -> Self {
        match value {
            DType::Bool(store) => match store {
                BoolStore::Native => BoolDType::Native,
                BoolStore::U8 => BoolDType::U8,
                BoolStore::U32 => BoolDType::U32,
            },
            // For compat BoolElem associated type
            DType::U8 => BoolDType::U8,
            DType::U32 => BoolDType::U32,
            _ => panic!("Expected bool data type, got {value:?}"),
        }
    }
}

impl From<BoolDType> for DType {
    fn from(value: BoolDType) -> Self {
        match value {
            BoolDType::Native => DType::Bool(BoolStore::Native),
            BoolDType::U8 => DType::Bool(BoolStore::U8),
            BoolDType::U32 => DType::Bool(BoolStore::U32),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finfo_f32() {
        let info = FloatDType::F32.finfo();
        assert_eq!(info.epsilon, f32::EPSILON as f64);
        assert_eq!(info.max, f32::MAX as f64);
        assert_eq!(info.min, f32::MIN as f64);
        assert_eq!(info.min_positive, f32::MIN_POSITIVE as f64);
    }

    #[test]
    fn finfo_f64() {
        let info = FloatDType::F64.finfo();
        assert_eq!(info.epsilon, f64::EPSILON);
        assert_eq!(info.max, f64::MAX);
        assert_eq!(info.min, f64::MIN);
        assert_eq!(info.min_positive, f64::MIN_POSITIVE);
    }

    #[test]
    fn finfo_f16() {
        let info = FloatDType::F16.finfo();
        assert_eq!(info.epsilon, f16::EPSILON.to_f64_const());
        assert!(info.epsilon > 0.0);
        assert!(info.min_positive > 0.0);
        // f16 epsilon is much larger than f32
        assert!(info.epsilon > FloatDType::F32.finfo().epsilon);
    }

    #[test]
    fn finfo_bf16() {
        let info = FloatDType::BF16.finfo();
        assert_eq!(info.epsilon, bf16::EPSILON.to_f64_const());
        assert!(info.epsilon > 0.0);
        assert!(info.min_positive > 0.0);
        // bf16 epsilon is larger than f32 (fewer mantissa bits)
        assert!(info.epsilon > FloatDType::F32.finfo().epsilon);
    }

    #[test]
    fn finfo_flex32_uses_f16_limits() {
        let flex = FloatDType::Flex32.finfo();
        let f16_info = FloatDType::F16.finfo();
        assert_eq!(flex.epsilon, f16_info.epsilon);
        assert_eq!(flex.min_positive, f16_info.min_positive);
    }

    #[test]
    fn dtype_finfo_delegates_to_float_dtype() {
        assert_eq!(DType::F32.finfo(), Some(FloatDType::F32.finfo()));
        assert_eq!(DType::F64.finfo(), Some(FloatDType::F64.finfo()));
        assert_eq!(DType::F16.finfo(), Some(FloatDType::F16.finfo()));
        assert_eq!(DType::BF16.finfo(), Some(FloatDType::BF16.finfo()));
        assert_eq!(DType::Flex32.finfo(), Some(FloatDType::Flex32.finfo()));
    }

    #[test]
    fn dtype_finfo_returns_none_for_non_float() {
        assert!(DType::I32.finfo().is_none());
        assert!(DType::U8.finfo().is_none());
        assert!(DType::Bool(BoolStore::Native).finfo().is_none());
    }

    #[test]
    fn finfo_invariants() {
        for dtype in [
            FloatDType::F64,
            FloatDType::F32,
            FloatDType::F16,
            FloatDType::BF16,
            FloatDType::Flex32,
        ] {
            let info = dtype.finfo();
            assert!(info.epsilon > 0.0, "{dtype:?}: epsilon must be positive");
            assert!(
                info.min_positive > 0.0,
                "{dtype:?}: min_positive must be positive"
            );
            assert!(info.max > 0.0, "{dtype:?}: max must be positive");
            assert!(info.min < 0.0, "{dtype:?}: min must be negative");
            assert!(
                info.max > info.min_positive,
                "{dtype:?}: max > min_positive"
            );
        }
    }
}
