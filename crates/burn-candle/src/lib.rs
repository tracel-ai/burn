#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(unused)] // TODO remove when backend filled
#![deprecated(
    since = "0.21.0-pre.1",
    note = "burn-candle is deprecated and will be removed in a future release. Use burn-cubecl (CUDA/ROCm/Vulkan/Metal/WebGPU), burn-ndarray, or burn-tch instead."
)]

//! Burn Candle Backend
//!
//! **Deprecated:** This backend is deprecated and will be removed in a future release.
//! Please migrate to one of the actively maintained backends:
//! - CubeCL backends (CUDA, ROCm, Vulkan, Metal, WebGPU) for GPU acceleration
//! - NdArray for portable CPU execution
//! - LibTorch (`burn-tch`) for a mature CPU/GPU backend

#[macro_use]
extern crate derive_new;

mod backend;
mod element;
mod ops;
mod tensor;

pub use backend::*;
pub use element::*;
pub use tensor::*;
