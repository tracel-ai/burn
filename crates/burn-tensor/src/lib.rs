#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! This library provides the core abstractions required to run tensor operations with Burn.
//! `Tensor`s are generic over the backend to allow users to perform operations using different `Backend` implementations.
//! Burn's tensors also support auto-differentiation thanks to the `AutodiffBackend` trait.

#[macro_use]
extern crate derive_new;

extern crate alloc;

mod tensor;

#[cfg(feature = "export_tests")]
#[allow(missing_docs)]
pub mod tests;

#[cfg(feature = "export_tests")]
// Re-export the might_panic proc macro for easy access
pub use burn_tensor_testgen::might_panic;

pub use half::{bf16, f16};
pub(crate) use tensor::check::macros::check;
pub use tensor::*;

pub use burn_common::stream_id::StreamId;

pub use burn_common::reader::*; // Useful so that backends don't have to add `burn_common` as a dependency.

#[cfg(feature = "cubecl")]
pub use cubecl::flex32;

#[cfg(feature = "cubecl")]
mod cube {
    use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, UIntKind};
    use cubecl_quant::scheme::QuantScheme;

    use crate::quantization::{QuantStore, QuantValue};

    impl From<crate::DType> for cubecl::ir::ElemType {
        fn from(dtype: crate::DType) -> Self {
            match dtype {
                crate::DType::F64 => ElemType::Float(FloatKind::F64),
                crate::DType::F32 => ElemType::Float(FloatKind::F32),
                crate::DType::Flex32 => ElemType::Float(FloatKind::Flex32),
                crate::DType::F16 => ElemType::Float(FloatKind::F16),
                crate::DType::BF16 => ElemType::Float(FloatKind::BF16),
                crate::DType::I64 => ElemType::Int(IntKind::I64),
                crate::DType::I32 => ElemType::Int(IntKind::I32),
                crate::DType::I16 => ElemType::Int(IntKind::I16),
                crate::DType::I8 => ElemType::Int(IntKind::I8),
                crate::DType::U64 => ElemType::UInt(UIntKind::U64),
                crate::DType::U32 => ElemType::UInt(UIntKind::U32),
                crate::DType::U16 => ElemType::UInt(UIntKind::U16),
                crate::DType::U8 => ElemType::UInt(UIntKind::U8),
                crate::DType::Bool => ElemType::Bool,
                crate::DType::QFloat(scheme) => match scheme.store {
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

    impl From<crate::DType> for cubecl::ir::StorageType {
        fn from(dtype: crate::DType) -> cubecl::ir::StorageType {
            match dtype {
                crate::DType::QFloat(QuantScheme {
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

#[cfg(feature = "cubecl-wgpu")]
mod cube_wgpu {
    use crate::backend::DeviceOps;
    use cubecl::wgpu::WgpuDevice;

    impl DeviceOps for WgpuDevice {}
}

#[cfg(feature = "cubecl-cuda")]
mod cube_cuda {
    use crate::backend::DeviceOps;
    use cubecl::cuda::CudaDevice;

    impl DeviceOps for CudaDevice {}
}

#[cfg(all(feature = "cubecl-cpu", target_os = "linux"))]
mod cube_cpu {
    use crate::backend::DeviceOps;
    use cubecl::cpu::CpuDevice;

    impl DeviceOps for CpuDevice {}
}

#[cfg(feature = "cubecl-hip")]
mod cube_hip {
    use crate::backend::DeviceOps;
    use cubecl::hip::AmdDevice;

    impl DeviceOps for AmdDevice {}
}
