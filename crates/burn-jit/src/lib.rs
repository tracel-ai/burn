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

use burn_tensor::backend::{DeviceId, DeviceOps};
use cubecl::{compute::CubeTask, Feature, Runtime};
pub use element::{BoolElement, FloatElement, IntElement, JitElement};

mod backend;

pub use backend::*;

// Re-export cubecl.
pub use cubecl;

mod tune_key;
pub use tune_key::JitAutotuneKey;

#[cfg(any(feature = "fusion", test))]
/// Module for interacting with fusion
pub mod fusion;

#[cfg(feature = "template")]
/// Module for compiling custom non-jit kernels
pub mod template;

#[cfg(feature = "export_tests")]
pub mod tests;

/// Just-in-Time runtime extending the [cube runtime](Runtime).
pub trait CubeRuntime: Runtime<Device = Self::JitDevice, Server = Self::JitServer> {
    /// The device that should also implement [burn_tensor::backend::DeviceOps].
    type JitDevice: burn_tensor::backend::DeviceOps;
    /// The cube server with the [JitAutotuneKey].
    type JitServer: cubecl::server::ComputeServer<
        Kernel = Box<dyn CubeTask<Self::Compiler>>,
        Feature = Feature,
    >;
}

/// ID used to identify a Just-in-Time environment.
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct JitTuneId {
    device: DeviceId,
    name: &'static str,
}

impl JitTuneId {
    /// Create a new ID.
    pub fn new<R: CubeRuntime>(device: &R::Device) -> Self {
        Self {
            device: DeviceOps::id(device),
            name: R::name(),
        }
    }
}

impl core::fmt::Display for JitTuneId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "device-{}-{}-{}",
            self.device.type_id, self.device.index_id, self.name
        ))
    }
}
