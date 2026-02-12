#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![recursion_limit = "138"]

//! Burn multi-backend dispatch.

#[cfg(not(any(
    feature = "cpu",
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan",
    feature = "webgpu",
    feature = "ndarray",
    feature = "tch",
)))]
compile_error!("At least one backend feature must be enabled.");

#[cfg(any(
    all(feature = "vulkan", feature = "metal"),
    all(feature = "vulkan", feature = "webgpu"),
    all(feature = "metal", feature = "webgpu")
))]
compile_error!("Only one wgpu runtime feature can be enabled at once.");

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
    #[cfg(feature = "cpu")]
    pub use burn_cpu::{Cpu, CpuDevice};
    #[cfg(feature = "cuda")]
    pub use burn_cuda::{Cuda, CudaDevice};
    #[cfg(feature = "rocm")]
    pub use burn_rocm::{Rocm, RocmDevice};
    #[cfg(feature = "metal")]
    pub use burn_wgpu::Metal;
    #[cfg(feature = "vulkan")]
    pub use burn_wgpu::Vulkan;
    #[cfg(feature = "webgpu")]
    pub use burn_wgpu::WebGpu;
    #[cfg(feature = "wgpu")]
    pub use burn_wgpu::WgpuDevice;

    #[cfg(feature = "ndarray")]
    pub use burn_ndarray::{NdArray, NdArrayDevice};
    #[cfg(feature = "tch")]
    pub use burn_tch::{LibTorch, LibTorchDevice};
}
