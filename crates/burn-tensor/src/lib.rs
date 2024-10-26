#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
// Allow deprecated `Data` and `DataSerialize`
#![allow(deprecated)]

//! This library provides multiple tensor implementations hidden behind an easy to use API
//! that supports reverse mode automatic differentiation.

#[macro_use]
extern crate derive_new;

extern crate alloc;

mod tensor;

/// Burn Tensor representaton
#[cfg(feature = "repr")]
pub mod repr;

#[cfg(feature = "export_tests")]
#[allow(missing_docs)]
mod tests;

pub use half::{bf16, f16};
pub(crate) use tensor::check::macros::check;
pub use tensor::*;

pub use burn_common::reader::*; // Useful so that backends don't have to add `burn_common` as a dependency.

#[cfg(feature = "cubecl")]
mod cube {
    use cubecl::ir::{Elem, FloatKind, IntKind, UIntKind};

    impl From<crate::DType> for cubecl::ir::Elem {
        fn from(dtype: crate::DType) -> Self {
            match dtype {
                crate::DType::F64 => Elem::Float(FloatKind::F64),
                crate::DType::F32 => Elem::Float(FloatKind::F32),
                crate::DType::F16 => Elem::Float(FloatKind::F16),
                crate::DType::BF16 => Elem::Float(FloatKind::BF16),
                crate::DType::I64 => Elem::Int(IntKind::I64),
                crate::DType::I32 => Elem::Int(IntKind::I32),
                crate::DType::I16 => Elem::Int(IntKind::I16),
                crate::DType::I8 => Elem::Int(IntKind::I8),
                crate::DType::U64 => Elem::UInt(UIntKind::U64),
                crate::DType::U32 => Elem::UInt(UIntKind::U32),
                crate::DType::U16 => Elem::UInt(UIntKind::U16),
                crate::DType::U8 => Elem::UInt(UIntKind::U8),
                crate::DType::Bool => Elem::Bool,
                crate::DType::QFloat(_) => panic!("quantized type is not supported yet."),
            }
        }
    }
}

#[cfg(feature = "cubecl-wgpu")]
mod cube_wgpu {
    use crate::backend::{DeviceId, DeviceOps};
    use cubecl::wgpu::WgpuDevice;

    impl DeviceOps for WgpuDevice {
        fn id(&self) -> DeviceId {
            match self {
                WgpuDevice::DiscreteGpu(index) => DeviceId::new(0, *index as u32),
                WgpuDevice::IntegratedGpu(index) => DeviceId::new(1, *index as u32),
                WgpuDevice::VirtualGpu(index) => DeviceId::new(2, *index as u32),
                WgpuDevice::Cpu => DeviceId::new(3, 0),
                WgpuDevice::BestAvailable => DeviceId::new(4, 0),
                // For an existing device, use the 64 bit wgpu device ID as the burn DeviceID.
                // We're only storing 32 bits, so wrap the the 64 bit value to 32 bits. This
                // might collide - but a 1 in 4 billion chance seems ok given there's only a few
                // devices in flight at any time.
                WgpuDevice::Existing(id) => {
                    DeviceId::new(5, (id.inner() % (u32::MAX as u64)) as u32)
                }
            }
        }
    }
}

#[cfg(feature = "cubecl-cuda")]
mod cube_cuda {
    use crate::backend::{DeviceId, DeviceOps};
    use cubecl::cuda::CudaDevice;

    impl DeviceOps for CudaDevice {
        fn id(&self) -> DeviceId {
            DeviceId::new(0, self.index as u32)
        }
    }
}
