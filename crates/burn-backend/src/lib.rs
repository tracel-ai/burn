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
    AllocationProperty, BoolDType, BoolStore, Bytes, DType, DataError, DeviceHandle, Distribution,
    DistributionSampler, DistributionSamplerKind, Element, ElementConversion, ElementEq,
    ElementOrdered, ElementRandom, FloatDType, IntDType, Scalar, TensorData, Tolerance, bf16,
    distribution, element, f16, stream_id::StreamId,
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
