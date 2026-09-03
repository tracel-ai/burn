#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

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
