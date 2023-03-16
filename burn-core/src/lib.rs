#![cfg_attr(not(feature = "std"), no_std)]

#[macro_use]
extern crate derive_new;

pub mod config;

#[cfg(feature = "std")]
pub mod data;

#[cfg(feature = "std")]
pub mod optim;

pub mod module;
pub mod nn;
pub mod tensor;

extern crate alloc;

#[cfg(all(test, not(feature = "test-tch")))]
pub type TestBackend = burn_ndarray::NdArrayBackend<f32>;

#[cfg(all(test, feature = "test-tch"))]
pub type TestBackend = burn_tch::TchBackend<f32>;

#[cfg(feature = "std")]
#[cfg(test)]
pub type TestADBackend = burn_autodiff::ADBackendDecorator<TestBackend>;
