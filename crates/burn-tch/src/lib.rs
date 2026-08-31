#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::single_range_in_vec_init)]
#![deprecated(
    since = "0.22.0",
    note = "burn-tch is deprecated and will be removed in a future release. For GPU acceleration use a CubeCL backend: burn-cuda (NVIDIA), burn-rocm (AMD), or burn-wgpu (Metal, Vulkan, WebGPU). For CPU execution use burn-cpu (CubeCL) or burn-flex (pure Rust)."
)]

//! Burn Tch Backend
//!
//! **Deprecated:** This backend is deprecated and will be removed in a future release.
//! Please migrate to one of the actively maintained backends:
//! - CubeCL GPU backends: `burn-cuda` (NVIDIA), `burn-rocm` (AMD), `burn-wgpu` (Metal, Vulkan,
//!   WebGPU)
//! - CPU backends: `burn-cpu` (CubeCL) or `burn-flex` (pure Rust, std/no_std/WASM)

mod backend;
mod element;
mod ops;
mod tensor;

pub use backend::*;
pub use element::*;
pub use tensor::*;
