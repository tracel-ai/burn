//! Vision ops for burn, with GPU acceleration where possible.
//!
//! # Operations
//! Operation names are based on `opencv` wherever applicable.
//!
//! Currently implemented are:
//! - `connected_components`
//! - `connected_components_with_stats`
//!

#![warn(missing_docs)]

extern crate alloc;

/// Backend implementations for JIT and CPU
pub mod backends;
mod ops;
mod tensor;

#[cfg(feature = "export-tests")]
#[allow(missing_docs)]
mod tests;

pub use ops::*;
pub use tensor::*;
