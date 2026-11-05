#![cfg_attr(docsrs, feature(doc_cfg))]

//! The ROCm (HIP) runtime for [Burn](https://github.com/tracel-ai/burn)'s CubeCL backend, for AMD
//! GPUs on Linux.
//!
//! Applications enable Burn's `rocm` feature and create a device with `Device::rocm(index)`;
//! they do not need this crate directly. [`Rocm`] is the backend type under the name of this
//! runtime: every CubeCL runtime shares the same backend, and a tensor's device says which one it
//! runs on.
//!
//! The HIP libraries are loaded at runtime, so building does not require ROCm to be installed.
//!
//! # Feature flags
//!
//! - `fusion` (default): kernel fusion.
//! - `autotune`: benchmark kernel variants at runtime and keep the fastest.
//! - `std`: standard library support.
//! - `tracing`: instrument operations with the `tracing` crate.
extern crate alloc;

pub use cubecl::hip::AmdDevice as RocmDevice;

/// The cubecl backend, under the name of the runtime this crate compiles in.
/// Every cubecl backend is the same type — a tensor's device is what says which
/// runtime it runs on.
pub type Rocm = burn_cubecl::Cube;
