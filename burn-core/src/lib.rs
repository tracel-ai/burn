#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! The core crate of Burn.

#[macro_use]
extern crate derive_new;

/// The configuration module.
pub mod config;

/// Data module.
#[cfg(feature = "std")]
pub mod data;

/// Optimizer module.
#[cfg(feature = "std")]
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

extern crate alloc;

#[cfg(all(
    test,
    not(feature = "test-tch"),
    not(feature = "test-wgpu"),
    not(feature = "test-candle")
))]
pub type TestBackend = burn_ndarray::NdArrayBackend<f32>;

#[cfg(all(test, feature = "test-tch"))]
pub type TestBackend = burn_tch::TchBackend<f32>;

#[cfg(all(test, feature = "test-candle"))]
pub type TestBackend = burn_candle::CandleBackend<f32, u32>;

#[cfg(all(test, feature = "test-wgpu", not(target_os = "macos")))]
pub type TestBackend = burn_wgpu::WgpuBackend<burn_wgpu::Vulkan, f32, i32>;

#[cfg(all(test, feature = "test-wgpu", target_os = "macos"))]
pub type TestBackend = burn_wgpu::WgpuBackend<burn_wgpu::Metal, f32, i32>;

#[cfg(feature = "std")]
#[cfg(test)]
pub type TestADBackend = burn_autodiff::ADBackendDecorator<TestBackend>;

/// Type alias for the learning rate.
///
/// LearningRate also implements [learning rate scheduler](crate::lr_scheduler::LRScheduler) so it
/// can be used for constant learning rate.
pub type LearningRate = f64; // We could potentially change the type.
