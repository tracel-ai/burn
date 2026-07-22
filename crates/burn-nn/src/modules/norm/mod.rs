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
use burn_core as burn;

use burn::tensor::{DType, FloatDType};

/// The dtype a normalization must accumulate its statistics in, or `None` when
/// the input's own dtype is already wide enough.
///
/// Normalizing squares its input, and `f16` cannot hold the result: its 5-bit
/// exponent tops out at 65 504, so an activation past ~256 squares to `inf` and
/// the division that follows yields `NaN`. That is not a corner case for a
/// diffusion U-Net, whose activations routinely reach the hundreds — it turns
/// the generated image blank. Torch widens the reduction to f32 for the same
/// reason, which is why fp16 U-Nets work there.
///
/// `bf16` is deliberately left alone: it carries f32's 8-bit exponent, so it
/// does not overflow, and widening it would cost a cast for no correctness gain.
pub(crate) fn accumulation_dtype(input: DType) -> Option<FloatDType> {
    matches!(input, DType::F16).then_some(FloatDType::F32)
}

pub(crate) mod batch;
pub(crate) mod group;
pub(crate) mod instance;
pub(crate) mod layer;
pub(crate) mod local_response;
pub(crate) mod rms;

mod normalization_wrapper;

pub use batch::*;
pub use group::*;
pub use instance::*;
pub use layer::*;
pub use local_response::*;
pub use normalization_wrapper::*;
pub use rms::*;
