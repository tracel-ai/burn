#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! This library provides multiple tensor implementations hidden behind an easy to use API
//! that supports reverse mode automatic differentiation.

#[macro_use]
extern crate derive_new;

extern crate alloc;

mod tensor;

#[cfg(feature = "export_tests")]
#[allow(missing_docs)]
mod tests;

pub use half::{bf16, f16};
pub use tensor::*;

pub use burn_common::reader::Reader; // Useful so that backends don't have to add `burn_common` as
                                     // a dependency so that they can implement the traits.

#[cfg(feature = "benchmark")]
/// This module provides benchmark utilities for easily and reliably run
/// benches on any function that is generic over a backend.
///
/// This can be useful to compare backends on  inference or training speed
/// for your models.
pub mod benchmark;
