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
mod base;
mod ops;
mod tensor;
mod transform;

pub use base::*;
pub use ops::*;
pub use tensor::*;
pub use transform::*;

/// Module for vision/image utilities
pub mod utils;

pub use backends::{KernelShape, create_structuring_element};
