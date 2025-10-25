//! # Normalization Layers
//!
//! Users who wish to provide an abstraction over swappable normalization
//! layers can use the [`Normalization`] wrapper, with support for:
//! * [`Normalization::Batch`] - [`BatchNorm`]
//! * [`Normalization::Group`] - [`GroupNorm`]
//! * [`Normalization::Instance`] - [`InstanceNorm`]
//! * [`Normalization::Layer`] - [`LayerNorm`]
//! * [`Normalization::Rms`] - [`RmsNorm`]
//!
//! [`NormalizationConfig`] can be used as a generic normalization policy:
//! * Construct a config with arbitrary input features (we suggest `0`).
//! * Clone and match that config to the target input layer,
//!   using the [`NormalizationConfig::with_num_features()`] method.
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
