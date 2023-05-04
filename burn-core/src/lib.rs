#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate derive_new;

pub mod config;

#[cfg(feature = "std")]
pub mod data;

#[cfg(feature = "std")]
pub mod optim;

#[cfg(feature = "std")]
pub mod lr_scheduler;

pub mod grad_clipper;
pub mod module;
pub mod nn;
pub mod record;
pub mod tensor;

extern crate alloc;

#[cfg(all(test, not(feature = "test-tch")))]
pub type TestBackend = burn_ndarray::NdArrayBackend<f32>;

#[cfg(all(test, feature = "test-tch"))]
pub type TestBackend = burn_tch::TchBackend<f32>;

#[cfg(feature = "std")]
#[cfg(test)]
pub type TestADBackend = burn_autodiff::ADBackendDecorator<TestBackend>;

/// Type alias for the learning rate.
///
/// LearningRate also implements [learning rate scheduler](crate::lr_scheduler::LRScheduler) so it
/// can be used for constant learning rate.
pub type LearningRate = f64; // We could potentially change the type.
