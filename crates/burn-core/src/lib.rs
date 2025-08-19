#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![recursion_limit = "135"]

//! The core crate of Burn.

#[macro_use]
extern crate derive_new;

/// Re-export serde for proc macros.
pub use serde;

/// The configuration module.
pub mod config;

/// Data module.
#[cfg(feature = "std")]
pub mod data;

/// Optimizer module.
pub mod optim;

/// Learning rate scheduler module.
#[cfg(feature = "std")]
pub mod lr_scheduler;

/// Gradient clipping module.
pub mod grad_clipping;

/// Module for the neural network module.
pub mod module;

/// Neural network module.
pub mod nn;

/// Module for the recorder.
pub mod record;

/// Module for the tensor.
pub mod tensor;

/// Module for visual operations
#[cfg(feature = "vision")]
pub mod vision;

extern crate alloc;

/// Backend for test cases
#[cfg(all(
    test,
    not(feature = "test-tch"),
    not(feature = "test-wgpu"),
    not(feature = "test-cuda")
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

/// Backend for autodiff test cases
#[cfg(test)]
pub type TestAutodiffBackend = burn_autodiff::Autodiff<TestBackend>;

#[cfg(all(test, feature = "test-memory-checks"))]
mod tests {
    burn_fusion::memory_checks!();
}

/// Type alias for the learning rate.
///
/// LearningRate also implements [learning rate scheduler](crate::lr_scheduler::LrScheduler) so it
/// can be used for constant learning rate.
pub type LearningRate = f64; // We could potentially change the type.

pub mod prelude {
    //! Structs and macros used by most projects. Add `use
    //! burn::prelude::*` to your code to quickly get started with
    //! Burn.
    pub use crate::{
        config::Config,
        module::Module,
        nn,
        tensor::{
            Bool, Device, ElementConversion, Float, Int, RangesArg, Shape, Tensor, TensorData,
            backend::Backend, cast::ToElement, s,
        },
    };
}
