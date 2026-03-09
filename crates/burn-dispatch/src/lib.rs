#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![recursion_limit = "138"]

//! Burn multi-backend dispatch.
//!
//! # Available Backends
//!
//! The dispatch backend supports the following variants, each enabled via cargo features:
//!
//! | Backend    | Feature    | Description |
//! |------------|------------|-------------|
//! | `Cpu`      | `cpu`      | Rust CPU backend (MLIR + LLVM) |
//! | `Cuda`     | `cuda`     | NVIDIA CUDA backend |
//! | `Metal`    | `metal`    | Apple Metal backend via `wgpu` (MSL) |
//! | `Rocm`     | `rocm`     | AMD ROCm backend |
//! | `Vulkan`   | `vulkan`   | Vulkan backend via `wgpu` (SPIR-V) |
//! | `WebGpu`   | `webgpu`   | WebGPU backend via `wgpu` (WGSL) |
//! | `NdArray`  | `ndarray`  | Pure Rust CPU backend using `ndarray` |
//! | `LibTorch` | `tch`      | Libtorch backend via `tch` |
//! | `Autodiff` | `autodiff` | Autodiff-enabled backend (used in combination with any of the backends above) |
//!
//! **Note:** WGPU-based backends (`metal`, `vulkan`, `webgpu`) are mutually exclusive.
//! All other backends can be combined freely.
//!
//! ## WGPU Backend Exclusivity
//!
//! The WGPU-based backends (`metal`, `vulkan`, `webgpu`) are **mutually exclusive** due to
//! the current automatic compile, which can only select one target at a time.
//!
//! Enable only **one** of these features in your `Cargo.toml`:
//! - `metal`
//! - `vulkan`
//! - `webgpu`
//!
//! If multiple WGPU features are enabled, the build script will emit a warning and **disable all WGPU
//! backends** to prevent unintended behavior.

#[cfg(not(any(
    feature = "cpu",
    feature = "cuda",
    wgpu_metal,
    feature = "rocm",
    wgpu_vulkan,
    wgpu_webgpu,
    feature = "ndarray",
    feature = "tch",
)))]
compile_error!("At least one backend feature must be enabled.");

#[macro_use]
mod macros;

mod backend;
mod device;
mod ops;
mod tensor;

pub use backend::*;
pub use device::*;
pub use tensor::*;

extern crate alloc;

/// Backends and devices used.
pub(crate) mod backends {
    #[cfg(feature = "autodiff")]
    pub use burn_autodiff::Autodiff;

    #[cfg(feature = "cpu")]
    pub use burn_cpu::{Cpu, CpuDevice};
    #[cfg(feature = "cuda")]
    pub use burn_cuda::{Cuda, CudaDevice};
    #[cfg(feature = "rocm")]
    pub use burn_rocm::{Rocm, RocmDevice};
    #[cfg(wgpu_metal)]
    pub use burn_wgpu::Metal;
    #[cfg(wgpu_vulkan)]
    pub use burn_wgpu::Vulkan;
    #[cfg(wgpu_webgpu)]
    pub use burn_wgpu::WebGpu;
    #[cfg(any(wgpu_metal, wgpu_vulkan, wgpu_webgpu))]
    pub use burn_wgpu::WgpuDevice;

    #[cfg(feature = "ndarray")]
    pub use burn_ndarray::{NdArray, NdArrayDevice};
    #[cfg(feature = "tch")]
    pub use burn_tch::{LibTorch, LibTorchDevice};
}
