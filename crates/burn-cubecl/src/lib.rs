#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// TODO: remove when fixed in cubecl
#![allow(semicolon_in_expressions_from_non_local_macros)]

//! The Burn backend for every [CubeCL](https://github.com/tracel-ai/cubecl) runtime.
//!
//! [`CubeBackend`] implements Burn's tensor operations as CubeCL kernels, compiled just in time
//! for the device they run on. CUDA, ROCm, Metal, Vulkan, WebGPU, wgpu and the CubeCL CPU
//! runtime all share this one backend type: a tensor's [`CubeDevice`] says which runtime it
//! uses. [`Cube`] is the type dispatch uses, wrapped in `burn_fusion::Fusion` when the `fusion`
//! feature is on.
//!
//! Applications reach this backend through a `burn` feature such as `cuda`, `wgpu` or `cpu`
//! and a `Device` constructor; the runtime crates (`burn-cuda`, `burn-wgpu`, `burn-rocm`,
//! `burn-cpu`) are thin wrappers that select a runtime. Use this crate directly to write
//! custom kernels: [`kernel`] and [`ops`] hold the building blocks, and [`cubecl`] is
//! re-exported so kernels use the same CubeCL version.
//!
//! # Feature flags
//!
//! - `cuda`, `hip`, `wgpu`, `metal`, `vulkan`, `webgpu`, `cpu`: compile in a CubeCL runtime.
//! - `fusion`: kernel fusion through `burn-fusion`.
//! - `autotune`: benchmark kernel variants at runtime and keep the fastest.
//! - `fft`: FFT kernels.
//! - `template`: launch hand-written, non-JIT kernels (see [`template`]).
//! - `tracing`: instrument operations with the `tracing` crate.

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

/// The device a cube tensor lives on.
///
/// One type across every runtime: which runtime a tensor runs on is what its
/// device *says*, not what its type is.
pub use cubecl::Device as CubeDevice;

pub use cubecl::CubeTuneId;

/// The tensor backend for every cubecl runtime.
///
/// CUDA, ROCm, Metal, Vulkan, WebGPU, wgpu and the CPU runtime are all this one
/// type; which of them a tensor runs on is what its [`CubeDevice`] says. Fusion
/// wraps it when the `fusion` feature is on.
#[cfg(not(feature = "fusion"))]
pub type Cube = CubeBackend;

/// The tensor backend for every cubecl runtime, fusing operations across
/// streams. See [`CubeBackend`] for the unfused type.
#[cfg(feature = "fusion")]
pub type Cube = burn_fusion::Fusion<CubeBackend>;
