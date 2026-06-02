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
//! | `Wgpu`     | `webgpu`   | WebGPU backend via `wgpu` (WGSL) |
//! | `Flex`     | `flex`     | Pure Rust CPU backend using `burn-flex` |
//! | `NdArray`  | `ndarray`  | Pure Rust CPU backend using `ndarray` (legacy - prefer `flex`) |
//! | `LibTorch` | `tch`      | Libtorch backend via `tch` |
//! | `Autodiff` | `autodiff` | Autodiff-enabled backend (used in combination with any of the backends above) |
//!
//! **Note:** All backends, including the WGPU-based ones (`wgpu`, `metal`, `vulkan`, `webgpu`),
//! can be combined freely. Each enabled wgpu backend appears as its own
//! [`DispatchDevice`] variant.

#[macro_use]
mod macros;

/// Dispatch backend module.
pub mod backend;
/// Dispatch device module.
pub mod device;
mod ops;
/// Dispatch tensor module.
pub mod tensor;

/// Entry points for hosting a remote-execution server.
#[cfg(feature = "remote-server")]
pub mod remote_server;

pub use backend::*;
pub use device::*;
pub use tensor::*;

extern crate alloc;

/// Backends and devices used.
pub mod backends {
    #[cfg(feature = "autodiff")]
    pub use burn_autodiff as autodiff;
    #[cfg(feature = "autodiff")]
    pub use burn_autodiff::Autodiff; // re-export for extensions

    #[cfg(feature = "cpu")]
    pub use burn_cpu as cpu;
    #[cfg(feature = "cpu")]
    pub use burn_cpu::Cpu;
    #[cfg(feature = "cuda")]
    pub use burn_cuda as cuda;
    #[cfg(feature = "cuda")]
    pub use burn_cuda::Cuda;
    #[cfg(feature = "rocm")]
    pub use burn_rocm as rocm;
    #[cfg(feature = "rocm")]
    pub use burn_rocm::Rocm;
    #[cfg(feature = "wgpu")]
    pub use burn_wgpu as wgpu;
    #[cfg(feature = "metal")]
    pub use burn_wgpu::Metal;
    #[cfg(feature = "vulkan")]
    pub use burn_wgpu::Vulkan;
    #[cfg(feature = "webgpu")]
    pub use burn_wgpu::WebGpu;
    #[cfg(feature = "wgpu")]
    pub use burn_wgpu::Wgpu;

    #[cfg(any(feature = "flex", default_backend))]
    pub use burn_flex as flex;
    #[cfg(any(feature = "flex", default_backend))]
    pub use burn_flex::Flex;
    #[cfg(feature = "ndarray")]
    pub use burn_ndarray as ndarray;
    #[cfg(feature = "ndarray")]
    pub use burn_ndarray::NdArray;
    #[cfg(feature = "tch")]
    pub use burn_tch as libtorch;
    #[cfg(feature = "tch")]
    pub use burn_tch::LibTorch;

    #[cfg(feature = "remote")]
    pub use burn_remote as remote;
    #[cfg(feature = "remote")]
    pub use burn_remote::RemoteBackend as Remote;

    pub use super::devices::*;
}

// Re-export devices

/// Backend devices.
pub mod devices {
    #[cfg(feature = "cpu")]
    pub use burn_cpu::CpuDevice;
    #[cfg(feature = "cuda")]
    pub use burn_cuda::CudaDevice;
    #[cfg(feature = "rocm")]
    pub use burn_rocm::RocmDevice;
    #[cfg(feature = "wgpu")]
    pub use burn_wgpu::WgpuDevice;

    #[cfg(any(feature = "flex", default_backend))]
    pub use burn_flex::FlexDevice;
    #[cfg(feature = "ndarray")]
    pub use burn_ndarray::NdArrayDevice;
    #[cfg(feature = "tch")]
    pub use burn_tch::LibTorchDevice;

    #[cfg(feature = "remote")]
    pub use burn_remote::RemoteDevice;
}
