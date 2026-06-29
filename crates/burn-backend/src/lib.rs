#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! This library provides the core types that define how Burn tensor data is represented, stored, and interpreted.

#[macro_use]
extern crate derive_new;

extern crate alloc;

/// [`Backend`] trait and required types.
pub mod backend;
pub use backend::*;

// Re-exported types
pub use burn_std::reader::*; // Useful so that backends don't have to add `burn_std` as a dependency.
pub use burn_std::{
    AllocationProperty, BoolDType, BoolStore, Bytes, ComplexDType, ComplexScalar, DType, DataError,
    DeviceHandle, Distribution, DistributionSampler, DistributionSamplerKind, Element,
    ElementConversion, ElementEq, ElementOrdered, ElementRandom, FloatDType, IntDType, Scalar,
    SplitPolicy, TensorData, Tolerance, bf16, complex_utils, distribution, element, f16,
    stream_id::StreamId,
};

/// Shape definition.
pub mod shape {
    pub use burn_std::shape::*;
}
pub use shape::*;

/// Slice utilities.
pub mod slice {
    pub use burn_std::{s, slice::*};
}
pub use slice::*;

/// Indexing utilities.
pub mod indexing {
    pub use burn_std::indexing::*;
}
pub use indexing::*;

mod alias;
pub use alias::*;

/// Quantization data representation.
pub mod quantization;

/// CubeCL inter-operation helpers (gated by the `cubecl` feature).
///
/// Provides plain conversion functions between burn's [`DType`] and cubecl's
/// `ElemType` / `StorageType`. They are intentionally exposed as named
/// functions rather than `From`/`Into` impls so the cubecl type tree does not
/// leak into `burn-std`'s public surface.
#[cfg(feature = "cubecl")]
pub mod cubecl;

#[cfg(any(
    feature = "cubecl-wgpu",
    feature = "cubecl-metal",
    feature = "cubecl-vulkan",
    feature = "cubecl-webgpu"
))]
mod cube_wgpu {
    use crate::backend::DeviceOps;
    use burn_std::{BoolStore, DType, DeviceSettings};
    use cubecl::wgpu::WgpuDevice;

    impl DeviceOps for WgpuDevice {
        #[cfg(not(any(feature = "cubecl-metal", feature = "cubecl-vulkan")))]
        fn defaults(&self) -> DeviceSettings {
            DeviceSettings::new(
                DType::F32,
                DType::I32,
                DType::Bool(BoolStore::U32),
                None,
                Default::default(),
            )
        }

        #[cfg(any(feature = "cubecl-metal", feature = "cubecl-vulkan"))]
        fn defaults(&self) -> DeviceSettings {
            DeviceSettings::new(
                DType::F32,
                DType::I32,
                DType::Bool(BoolStore::U8),
                None,
                Default::default(),
            )
        }
    }
}

#[cfg(feature = "cubecl-cuda")]
mod cube_cuda {
    use crate::backend::DeviceOps;
    use burn_std::{BoolStore, DType, DeviceSettings};
    use cubecl::cuda::CudaDevice;

    impl DeviceOps for CudaDevice {
        fn defaults(&self) -> DeviceSettings {
            DeviceSettings::new(
                DType::F32,
                DType::I32,
                DType::Bool(BoolStore::U8),
                None,
                Default::default(),
            )
        }
    }
}

#[cfg(feature = "cubecl-cpu")]
mod cube_cpu {
    use crate::backend::DeviceOps;
    use burn_std::{BoolStore, DType, DeviceSettings};
    use cubecl::cpu::CpuDevice;

    impl DeviceOps for CpuDevice {
        fn defaults(&self) -> DeviceSettings {
            DeviceSettings::new(
                DType::F32,
                DType::I32,
                DType::Bool(BoolStore::U8),
                None,
                Default::default(),
            )
        }
    }
}

#[cfg(feature = "cubecl-hip")]
mod cube_hip {
    use crate::backend::DeviceOps;
    use burn_std::{BoolStore, DType, DeviceSettings};
    use cubecl::hip::AmdDevice;

    impl DeviceOps for AmdDevice {
        fn defaults(&self) -> DeviceSettings {
            DeviceSettings::new(
                DType::F32,
                DType::I32,
                DType::Bool(BoolStore::U8),
                None,
                Default::default(),
            )
        }
    }
}

/// Convenience macro to link to the `burn-tensor` docs for this crate version.
///
/// Usage:
/// ```rust,ignore
/// # use burn_backend::doc_tensor;
/// doc_tensor!();        // Links to `Tensor` struct
/// doc_tensor!("zeros"); // Links to `Tensor::zeros` method
/// ```
#[macro_export]
macro_rules! doc_tensor {
    () => {
        concat!(
            "[`Tensor`](https://docs.rs/burn-tensor/",
            env!("CARGO_PKG_VERSION"),
            "/burn_tensor/struct.Tensor.html)"
        )
    };

    ($method:literal) => {
        concat!(
            "[`Tensor::",
            $method,
            "`](",
            "https://docs.rs/burn-tensor/",
            env!("CARGO_PKG_VERSION"),
            "/burn_tensor/struct.Tensor.html#method.",
            $method,
            ")"
        )
    };
}
