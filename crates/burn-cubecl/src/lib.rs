#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! Burn JIT Backend

#[macro_use]
extern crate derive_new;
extern crate alloc;

/// Utilities for implementing JIT kernels
pub mod ops;

/// Kernel module
pub mod kernel;
/// Tensor module.
pub mod tensor;

/// Elements for JIT backend
pub mod element;

use cubecl::{compute::CubeTask, Feature, Runtime};
pub use element::{BoolElement, CubeElement, FloatElement, IntElement};

mod backend;

pub use backend::*;

// Re-export cubecl.
pub use cubecl;

mod tune_key;
pub use tune_key::CubeAutotuneKey;

#[cfg(any(feature = "fusion", test))]
/// Module for interacting with fusion
pub mod fusion;

#[cfg(feature = "template")]
/// Module for compiling custom non-jit kernels
pub mod template;

#[cfg(feature = "export_tests")]
pub mod tests;

/// Just-in-Time runtime extending the [cube runtime](Runtime).
pub trait CubeRuntime: Runtime<Device = Self::CubeDevice, Server = Self::CubeServer> {
    /// The device that should also implement [burn_tensor::backend::DeviceOps].
    type CubeDevice: burn_tensor::backend::DeviceOps;
    /// The cube server with the [CubeAutotuneKey].
    type CubeServer: cubecl::server::ComputeServer<
        Kernel = Box<dyn CubeTask<Self::Compiler>>,
        Feature = Feature,
    >;
}

pub use cubecl::CubeTuneId;
