#![warn(missing_docs)]

//! Burn JIT Backend

#[macro_use]
extern crate derive_new;
extern crate alloc;

mod ops;

/// Kernel module
pub mod kernel;
/// Tensor module.
pub mod tensor;

pub(crate) mod tune;

/// Elements for JIT backend
pub mod element;

use burn_cube::{compute::CubeTask, Runtime};
pub use element::{FloatElement, IntElement, JitElement};

mod backend;
mod bridge;

pub use backend::*;
pub use bridge::*;

mod tune_key;
pub use tune_key::JitAutotuneKey;

#[cfg(any(feature = "fusion", test))]
mod fusion;

#[cfg(feature = "template")]
/// Module for compiling custom non-jit kernels
pub mod template;

#[cfg(feature = "export_tests")]
pub mod tests;

/// Just-in-Time runtime extending the [cube runtime](Runtime).
pub trait JitRuntime: Runtime<Device = Self::JitDevice, Server = Self::JitServer> {
    /// The device that should also implement [DeviceOps](burn_tensor::backend::DeviceOps).
    type JitDevice: burn_tensor::backend::DeviceOps;
    /// The cube server with the [JitAutotuneKey].
    type JitServer: burn_compute::server::ComputeServer<
        AutotuneKey = JitAutotuneKey,
        Kernel = Box<dyn CubeTask>,
    >;
}
