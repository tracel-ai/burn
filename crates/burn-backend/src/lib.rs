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

    impl DeviceOps for Device {
        fn defaults(&self) -> DeviceSettings {
            // wgsl has no 8-bit type to store a bool in, so under the portable
            // compiler a bool costs a word. Compiling straight to Metal or
            // SPIR-V, and on every other runtime, a byte will do.
            let bool_store = match self.runtime() {
                RuntimeId::Wgpu
                    if !cfg!(any(feature = "cubecl-metal", feature = "cubecl-vulkan")) =>
                {
                    BoolStore::U32
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
