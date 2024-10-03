#![warn(missing_docs)]

//! A library for training neural networks using the burn crate.

#[macro_use]
extern crate derive_new;

/// The checkpoint module.
pub mod checkpoint;

pub(crate) mod components;

/// Renderer modules to display metrics and training information.
pub mod renderer;
pub mod renderer_test;

/// The logger module.
pub mod logger;

/// The metric module.
pub mod metric;
pub mod metric_test;

mod learner;

pub use learner::*;

#[cfg(test)]
pub(crate) type TestBackend = burn_ndarray::NdArray<f32>;
