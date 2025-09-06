//! # Normalization Layers
//!
//! Users who wish to provide an abstraction over swappable normalization
//! layers can use the [`Normalization`] wrapper, with support for:
/// * [`Batch`] - [`BatchNorm`]
/// * [`Group`] - [`GroupNorm`]
/// * [`Instance`] - [`InstanceNorm`]
/// * [`Layer`] - [`LayerNorm`]
/// * [`Rms`] - [`RmsNorm`]
pub(crate) mod batch;
pub(crate) mod group;
pub(crate) mod instance;
pub(crate) mod layer;
pub(crate) mod rms;

mod normalization_wrapper;

pub use batch::*;
pub use group::*;
pub use instance::*;
pub use layer::*;
pub use normalization_wrapper::*;
pub use rms::*;
