//! Tensor data type.

use serde::{Deserialize, Serialize};

use crate::Shape;
use crate::tensor::quantization::{
    QPARAM_ALIGN, QuantParam, QuantScheme, QuantStore, QuantValue, params_shape,
};
use crate::{bf16, f16};

/// Returns the byte size of a quantization parameter type.
// TODO: Add `size_bytes()` method to `QuantParam` in cubecl and use it here.
const fn quant_param_size(param: QuantParam) -> usize {
    match param {
        QuantParam::F32 => core::mem::size_of::<f32>(),
        QuantParam::F16 | QuantParam::BF16 => core::mem::size_of::<f16>(),
        QuantParam::UE8M0 | QuantParam::UE4M3 => core::mem::size_of::<u8>(),
    }
}

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

    /// Returns the total byte size for tensor data with the given shape.
    ///
    /// For regular (non-quantized) types, this is simply `shape.product() * self.size()`.
    ///
    /// For quantized types (`QFloat`), this accounts for:
    /// - The quantized values (packed according to the quantization scheme)
    /// - Alignment padding (values are aligned to 4-byte boundary)
    /// - Quantization parameters (scale values appended to the data)
    pub fn data_bytes(&self, shape: &[usize]) -> usize {
        const BITS_PER_BYTE: usize = 8;

        let num_elements: usize = shape.iter().product();

        match self {
            DType::QFloat(scheme) => {
                // Calculate value bytes using scheme's packing information
                let num_storage_elements = num_elements.div_ceil(scheme.num_quants());
                let value_bytes =
                    num_storage_elements * (scheme.size_bits_stored() / BITS_PER_BYTE);

                // Calculate number of quantization parameters (scales)
                let num_params =
                    params_shape(&Shape::from(shape.to_vec()), scheme.level).num_elements();

                let aligned_value_bytes = value_bytes.div_ceil(QPARAM_ALIGN) * QPARAM_ALIGN;
                let scale_bytes = num_params * quant_param_size(scheme.param);

                aligned_value_bytes + scale_bytes
            }
            _ => num_elements * self.size(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantization::QuantLevel;

    #[test]
    fn test_data_bytes_regular_types() {
        // Test that data_bytes returns shape.product() * size() for regular types
        let shape = &[2, 3, 4]; // 24 elements

        assert_eq!(DType::F64.data_bytes(shape), 24 * 8);
        assert_eq!(DType::F32.data_bytes(shape), 24 * 4);
        assert_eq!(DType::F16.data_bytes(shape), 24 * 2);
        assert_eq!(DType::BF16.data_bytes(shape), 24 * 2);
        assert_eq!(DType::I64.data_bytes(shape), 24 * 8);
        assert_eq!(DType::I32.data_bytes(shape), 24 * 4);
        assert_eq!(DType::I16.data_bytes(shape), 24 * 2);
        assert_eq!(DType::I8.data_bytes(shape), 24 * 1);
        assert_eq!(DType::U64.data_bytes(shape), 24 * 8);
        assert_eq!(DType::U32.data_bytes(shape), 24 * 4);
        assert_eq!(DType::U16.data_bytes(shape), 24 * 2);
        assert_eq!(DType::U8.data_bytes(shape), 24 * 1);
        assert_eq!(DType::Bool.data_bytes(shape), 24 * 1);
    }

    #[test]
    fn test_data_bytes_quantized_tensor_level() {
        use crate::tensor::quantization::QuantParam;

        // Q8S with tensor-level quantization
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native)
            .with_level(QuantLevel::Tensor)
            .with_param(QuantParam::F32);
        let dtype = DType::QFloat(scheme);

        // Shape [512, 512] = 262144 elements
        // Values: 262144 bytes (1 byte per Q8S element)
        // Aligned: 262144 (already aligned to 4)
        // Scale: 4 bytes (1 f32 for tensor-level)
        // Total: 262144 + 4 = 262148
        let shape = &[512, 512];
        assert_eq!(dtype.data_bytes(shape), 262148);

        // Shape [5] = 5 elements
        // Values: 5 bytes
        // Aligned: 8 bytes (5 rounded up to multiple of 4)
        // Scale: 4 bytes
        // Total: 8 + 4 = 12
        let shape = &[5];
        assert_eq!(dtype.data_bytes(shape), 12);
    }

    #[test]
    fn test_data_bytes_quantized_block_level() {
        use crate::tensor::quantization::QuantParam;

        // Q8S with block-level quantization (block size 32)
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native)
            .with_level(QuantLevel::block([32]))
            .with_param(QuantParam::F32);
        let dtype = DType::QFloat(scheme);

        // Shape [128, 128] = 16384 elements
        // Values: 16384 bytes (1 byte per Q8S element)
        // Aligned: 16384 (already aligned to 4)
        // Block size [32] expands to [32, 1] for 2D tensor (last dim is 1)
        // Num blocks: ceil(128/32) * ceil(128/1) = 4 * 128 = 512
        // Scale: 512 * 4 = 2048 bytes
        // Total: 16384 + 2048 = 18432
        let shape = &[128, 128];
        assert_eq!(dtype.data_bytes(shape), 18432);

        // 1D tensor with block size [32]
        // Shape [128] = 128 elements
        // Values: 128 bytes
        // Aligned: 128 (already aligned)
        // Block size [32] => [32] for 1D
        // Num blocks: ceil(128/32) = 4
        // Scale: 4 * 4 = 16 bytes
        // Total: 128 + 16 = 144
        let shape = &[128];
        assert_eq!(dtype.data_bytes(shape), 144);
    }

    #[test]
    fn test_data_bytes_quantized_u32_store() {
        use crate::tensor::quantization::QuantParam;

        // Q8S with U32 store (values packed 4 per u32)
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::U32)
            .with_level(QuantLevel::Tensor)
            .with_param(QuantParam::F32);
        let dtype = DType::QFloat(scheme);

        // Shape [16] = 16 elements
        // Values: 16 elements / 4 per u32 = 4 u32s = 16 bytes
        // Aligned: 16 bytes (already aligned)
        // Scale: 4 bytes
        // Total: 16 + 4 = 20
        let shape = &[16];
        assert_eq!(dtype.data_bytes(shape), 20);

        // Shape [17] = 17 elements
        // Values: ceil(17/4) = 5 u32s = 20 bytes
        // Aligned: 20 bytes
        // Scale: 4 bytes
        // Total: 20 + 4 = 24
        let shape = &[17];
        assert_eq!(dtype.data_bytes(shape), 24);
    }
}
