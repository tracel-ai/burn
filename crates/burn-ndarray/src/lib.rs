#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deprecated(
    since = "0.22.0",
    note = "burn-ndarray is deprecated and will be removed in a future release. Use burn-flex for pure-Rust CPU execution (std, no_std, WebAssembly), or one of the CubeCL backends (burn-cuda, burn-rocm, burn-wgpu, burn-cpu) for GPU acceleration."
)]

//! Burn ndarray backend.
//!
//! **Deprecated:** This backend is deprecated and will be removed in a future release.
//! Please migrate to one of the actively maintained backends:
//! - Flex (`burn-flex`) for portable pure-Rust CPU execution (std, no_std, WASM)
//! - CubeCL backends (CUDA, ROCm, Vulkan, Metal, WebGPU) for GPU acceleration
//!
//! See [COMPARISON.md](https://github.com/tracel-ai/burn/blob/main/crates/burn-flex/COMPARISON.md)
//! for an operation-by-operation migration path.

#[cfg(any(
    feature = "blas-netlib",
    feature = "blas-openblas",
    feature = "blas-openblas-system",
))]
extern crate blas_src;

mod backend;
mod element;
mod ops;
mod parallel;
mod rand;
mod sharing;
mod storage;
mod tensor;

pub use backend::*;
pub use element::*;
pub(crate) use sharing::*;
pub(crate) use storage::*;
pub use tensor::*;

extern crate alloc;
