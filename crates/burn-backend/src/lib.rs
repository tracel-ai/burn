#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! This library provides the core types that define how Burn tensor data is represented, stored, and interpreted.

#[macro_use]
extern crate derive_new;

extern crate alloc;

mod data;
pub use data::*;

pub mod distribution;
pub use distribution::*;
pub mod element;
pub use element::*;

/// [`Backend`] trait and required types.
pub mod backend;
pub use backend::*;

/// Backend tensor primitives and operations.
pub mod tensor;

// Re-exported types
pub use burn_std::reader::*; // Useful so that backends don't have to add `burn_std` as a dependency.
pub use burn_std::{
    AllocationProperty, Bytes, DType, DeviceHandle, FloatDType, IntDType, bf16, f16,
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

/// Quantization data representation.
pub mod quantization {
    pub use crate::tensor::quantization::*;
    pub use burn_std::quantization::{
        BlockSize, QuantLevel, QuantMode, QuantParam, QuantPropagation, QuantScheme, QuantStore,
        QuantValue, QuantizedBytes,
    };
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

#[cfg(feature = "cubecl-cpu")]
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
