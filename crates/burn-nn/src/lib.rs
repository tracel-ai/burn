#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
// #![recursion_limit = "135"]

//! Burn neural network module.

// /// Re-export serde for proc macros.
// pub use serde;

// /// The configuration module.
// pub mod config;

/// Neural network module core.
pub mod module;

// /// Module for the recorder.
// pub mod record;

/// Loss module
pub mod loss;

/// Neural network modules implementations.
pub mod modules;

pub mod activation;
pub use activation::{
    gelu::*, glu::*, hard_sigmoid::*, leaky_relu::*, prelu::*, relu::*, sigmoid::*, swiglu::*,
    tanh::*,
};

mod initializer;
mod padding;
pub use initializer::*;
pub use padding::*;

extern crate alloc;

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

#[cfg(all(test, feature = "test-memory-checks"))]
mod tests {
    burn_fusion::memory_checks!();
}
