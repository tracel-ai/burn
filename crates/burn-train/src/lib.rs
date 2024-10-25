#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

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

mod learner;

pub use learner::*;

#[cfg(test)]
pub(crate) type TestBackend = burn_ndarray::NdArray<f32>;
