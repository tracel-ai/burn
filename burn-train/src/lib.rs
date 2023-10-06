#![warn(missing_docs)]

//! A library for training neural networks using the burn crate.

#[macro_use]
extern crate derive_new;

/// The checkpoint module.
pub mod checkpoint;

pub(crate) mod components;

/// The logger module.
pub mod logger;

/// The metric module.
pub mod metric;

mod callback;
mod learner;

pub use callback::*;
pub use learner::*;

#[cfg(test)]
pub(crate) type TestBackend = burn_ndarray::NdArrayBackend<f32>;
