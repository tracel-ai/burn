#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! # Burn PEFT - Parameter-Efficient Fine-Tuning
//!
//! This crate provides state-of-the-art parameter-efficient fine-tuning methods for Burn:
//!
//! - **LoRA** (Low-Rank Adaptation): Efficient fine-tuning via low-rank weight updates
//! - **DoRA** (Weight-Decomposed Low-Rank Adaptation): Extends LoRA by decomposing weights into magnitude and direction
//! - **QLoRA**: LoRA with quantized base weights for memory efficiency
//! - **QDoRA**: DoRA with quantized base weights
//!
//! ## Features
//!
//! - Drop-in replacement for `Linear` layers
//! - Exact mathematical semantics with numerical stability
//! - Support for merging adapters for inference
//! - Quantization-aware implementations (QLoRA/QDoRA)
//! - Configurable initialization strategies
//!
//! ## Usage
//!
//! ```rust,ignore
//! use burn_peft::{LoRAConfig, LoRALinear};
//! use burn::tensor::backend::Backend;
//!
//! // Create a LoRA-adapted linear layer
//! let config = LoRAConfig::new(512, 512)
//!     .with_rank(16)
//!     .with_alpha(32.0);
//!
//! let layer = config.init::<MyBackend>(&device);
//! let output = layer.forward(input);
//! ```

extern crate alloc;

mod ops;
pub use ops::*;

mod lora;
pub use lora::*;

mod dora;
pub use dora::*;

mod qlora;
pub use qlora::*;

mod qdora;
pub use qdora::*;

mod compose;
pub use compose::*;

mod lora_plus;
pub use lora_plus::*;

/// Backend for test cases
#[cfg(all(
    test,
    not(feature = "test-tch"),
    not(feature = "test-wgpu"),
    not(feature = "test-cuda"),
    not(feature = "test-rocm")
))]
pub type TestBackend = burn_ndarray::NdArray<f32>;

#[cfg(all(test, feature = "test-tch"))]
/// Backend for test cases
pub type TestBackend = burn_tch::LibTorch<f32>;

#[cfg(all(test, feature = "test-wgpu"))]
/// Backend for test cases
pub type TestBackend = burn_wgpu::Wgpu;

#[cfg(all(test, feature = "test-cuda"))]
/// Backend for test cases
pub type TestBackend = burn_cuda::Cuda;

#[cfg(all(test, feature = "test-rocm"))]
/// Backend for test cases
pub type TestBackend = burn_rocm::Rocm;

/// Backend for autodiff test cases
#[cfg(test)]
pub type TestAutodiffBackend = burn_autodiff::Autodiff<TestBackend>;
