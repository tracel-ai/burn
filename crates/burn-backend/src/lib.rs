#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! The contract between Burn's tensor API and the backends that execute it.
//!
//! Application code does not use this crate directly: it works with `burn::tensor::Tensor`
//! and `Device`, and dispatch picks a backend at runtime. This crate is for code below that
//! boundary, such as backend implementations, backend decorators and backend extensions.
//!
//! - [`BackendTypes`] names a backend's tensor primitives and device type.
//! - [`Backend`] and the operation traits in [`ops`] define every tensor operation a backend
//!   must implement. [`AutodiffBackend`] adds gradient support on top.
//! - [`DeviceOps`] describes a backend device and its default dtypes.
//! - [`TensorData`], [`DType`], [`Shape`] and the element traits describe tensor data
//!   independently of any backend.
//!
//! Backend operations report device failures as [`ExecutionError`] instead of panicking.
//!
//! # Feature flags
//!
//! - `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
//! - `cubecl`: conversions between Burn and CubeCL types.
//! - `cubecl-device` and the `cubecl-<runtime>` features: implement [`DeviceOps`] for
//!   `cubecl::Device`.
//! - `tracing`: instrument operations with the `tracing` crate.

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
    DistributionSampler, DistributionSamplerKind, Element, ElementAdd, ElementConversion,
    ElementEq, ElementOrdered, ElementRandom, FloatDType, IntDType, Scalar, SplitPolicy,
    TensorData, Tolerance, bf16, distribution, element, f16, stream::StreamId,
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

// Not gated on the `cubecl-*` runtime features: a build gets its cubecl runtime from whichever
// crate asked for one, which need not be this one — `burn-cubecl` compiles with no runtime feature
// of its own and still needs this impl. So the impl is its own feature, and a crate that needs
// `cubecl::Device` to be a burn device says so.
#[cfg(feature = "cubecl-device")]
mod cube_device {
    use crate::backend::DeviceOps;
    use burn_std::{BoolStore, DType, DeviceSettings};
    use cubecl::{Device, RuntimeId};
    use cubecl::{
        features::TypeUsage,
        ir::{ElemType, UIntKind},
    };

    impl DeviceOps for Device {
        fn defaults(&self) -> DeviceSettings {
            // Cargo features make native compilers available, but automatic devices can still
            // fall back to WGSL. Only use byte-sized bools when this device can store and convert
            // them; WGSL needs a word. Other runtimes continue to use a byte.
            let bool_store = match self.runtime() {
                RuntimeId::Wgpu => {
                    let usage = self
                        .client()
                        .properties()
                        .type_usage(ElemType::UInt(UIntKind::U8));
                    if usage.is_superset(TypeUsage::Buffer | TypeUsage::Conversion) {
                        BoolStore::U8
                    } else {
                        BoolStore::U32
                    }
                }
                _ => BoolStore::U8,
            };

            DeviceSettings::new(
                DType::F32,
                DType::I32,
                DType::Bool(bool_store),
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
