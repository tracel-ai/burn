#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![recursion_limit = "138"]
// Wiring up the deprecated `NdArray` and `LibTorch` backends is this crate's job, and the backend
// registry macros expand them into every dispatch impl, so the warnings land on `macros.rs` rather
// than on any site we could annotate individually. `allow(deprecated)` is a lint level scoped to
// this crate, and lint levels never propagate to dependents, so downstream code naming `NdArray` or
// `LibTorch` (directly or via our re-export) still gets the warning. The `cfg_attr` keeps this
// confined to the builds that enable them: neither `ndarray` nor `tch` is a default feature, so the
// default build that CI lints with `--deny warnings` retains full deprecation signal for every
// other dependency.
#![cfg_attr(any(feature = "ndarray", feature = "tch"), allow(deprecated))]

//! Burn multi-backend dispatch.
//!
//! # Available Backends
//!
//! The dispatch backend supports the following variants, each enabled via cargo features:
//!
//! | Backend    | Feature    | Description |
//! |------------|------------|-------------|
//! | `Cube`     | `cpu`, `cuda`, `metal`, `rocm`, `vulkan`, `webgpu`, `wgpu` | Every cubecl runtime. One backend: the features decide which runtimes are compiled in, and a tensor's device says which one it runs on |
//! | `Flex`     | `flex`     | Pure Rust CPU backend using `burn-flex` |
//! | `NdArray`  | `ndarray`  | Pure Rust CPU backend using `ndarray` (deprecated - use `flex`) |
//! | `LibTorch` | `tch`      | Libtorch backend via `tch` (deprecated - use a CubeCL backend) |
//! | `Autodiff` | `autodiff` | Autodiff-enabled backend (used in combination with any of the backends above) |
//!
//! **Note:** The features can be combined freely. The cubecl-backed ones all
//! select the same backend, so they share the one [`DispatchDevice::Cube`]
//! variant — enabling several compiles several runtimes in, and the device a
//! tensor carries is what picks between them.

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
    #[cfg(feature = "cuda")]
    pub use burn_cuda as cuda;
    #[cfg(feature = "rocm")]
    pub use burn_rocm as rocm;
    #[cfg(feature = "wgpu")]
    pub use burn_wgpu as wgpu;

    /// The cubecl backend: CUDA, ROCm, Metal, Vulkan, WebGPU, wgpu and the CPU
    /// runtime are all this one type, and a tensor's device says which of them
    /// it runs on. The features still decide which runtimes are compiled in.
    #[cfg(cube_backend)]
    pub use burn_cubecl::Cube;

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

    /// Public graph-capture API types.
    #[cfg(feature = "capture")]
    pub mod capture {
        pub use burn_capture::{
            CaptureBackend, CaptureError, CaptureScope, CapturedGraph, CompletedCaptureScope,
            TensorId,
        };
    }
    #[cfg(feature = "capture")]
    pub use burn_capture::CaptureBackend as Capture;

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

    /// The device every cubecl runtime shares; which runtime it names is a
    /// property of the value, not of its type, and [`RuntimeId`] is how that
    /// property is named.
    #[cfg(cube_backend)]
    pub use burn_cubecl::CubeDevice;
    #[cfg(cube_backend)]
    pub use burn_cubecl::cubecl::RuntimeId;
    #[cfg(any(feature = "flex", default_backend))]
    pub use burn_flex::FlexDevice;
    #[cfg(feature = "ndarray")]
    pub use burn_ndarray::NdArrayDevice;
    #[cfg(feature = "tch")]
    pub use burn_tch::LibTorchDevice;

    #[cfg(feature = "remote")]
    pub use burn_remote::RemoteDevice;

    #[cfg(feature = "remote")]
    pub use burn_remote::BURN_REMOTE_ALPN;
}
