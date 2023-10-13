#![warn(missing_docs)]

//! A library for training neural networks using the burn crate.

#[macro_use]
extern crate derive_new;

/// The checkpoint module.
pub mod checkpoint;

pub(crate) mod components;

/// Renderer modules to display metrics and training information.
pub mod renderer;

/// The logger module.
pub mod logger;

/// The metric module.
pub mod metric;

/// All information collected during training.
pub mod info;

mod collector;
mod learner;

pub use collector::*;
pub use learner::*;

#[cfg(test)]
pub(crate) type TestBackend = burn_ndarray::NdArrayBackend<f32>;
