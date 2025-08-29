//! # Normalization Layers
//!
//! Users who wish to provide an abstraction over swappable normalization
//! layers can use the [`Normalization`] wrapper, with support for:
/// * [`Batch`] - [`BatchNorm`]
/// * [`Group`] - [`GroupNorm`]
/// * [`Instance`] - [`InstanceNorm`]
/// * [`Layer`] - [`LayerNorm`]
/// * [`Rms`] - [`RmsNorm`]
mod batch;
mod group;
mod instance;
mod layer;
mod normalization_wrapper;
mod rms;

pub use batch::*;
pub use group::*;
pub use instance::*;
pub use layer::*;
pub use normalization_wrapper::*;
pub use rms::*;
