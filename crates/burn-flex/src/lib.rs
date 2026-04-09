#![cfg_attr(not(feature = "std"), no_std)]

//! # burn-flex
//!
//! A fast, portable CPU backend for [Burn](https://github.com/tracel-ai/burn).
//!
//! ## Features
//!
//! - Pure Rust (no C dependencies)
//! - f16/bf16 support
//! - SIMD acceleration via macerator (NEON, AVX2/AVX-512/SSE, SIMD128, scalar fallback)
//! - Zero-copy tensor views
//! - Thread-safe by design
//!
//! ## Usage
//!
//! ```ignore
//! use burn_flex::Flex;
//! use burn::tensor::Tensor;
//!
//! let tensor: Tensor<Flex, 2> = Tensor::from_data([[1.0, 2.0], [3.0, 4.0]], &Default::default());
//! ```

extern crate alloc;

#[cfg(all(not(target_has_atomic = "ptr"), not(feature = "critical-section")))]
compile_error!(
    "This target lacks atomic CAS support. Enable the `critical-section` feature: \
     burn-flex = { ..., features = [\"critical-section\"] }"
);

mod backend;
mod layout;
mod qtensor;
mod strided_index;
mod tensor;

#[doc(hidden)]
pub mod ops;

#[doc(hidden)]
pub mod simd;

pub use backend::{Flex, FlexDevice};
pub use layout::Layout;
pub use qtensor::FlexQTensor;
pub use tensor::FlexTensor;
