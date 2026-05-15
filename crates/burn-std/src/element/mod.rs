//! Traits and helpers for working with element types and conversions.

mod base;
mod scalar;

/// Tensor element casting.
pub mod cast;

pub use base::*;
pub use scalar::*;
