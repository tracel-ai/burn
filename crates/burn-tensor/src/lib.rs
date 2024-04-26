#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]

//! This library provides multiple tensor implementations hidden behind an easy to use API
//! that supports reverse mode automatic differentiation.

#[macro_use]
extern crate derive_new;

extern crate alloc;

mod tensor;

/// Burn Tensor representaton
#[cfg(feature = "repr")]
pub mod repr;

#[cfg(feature = "export_tests")]
#[allow(missing_docs)]
mod tests;

pub use half::{bf16, f16};
pub(crate) use tensor::check::macros::check;
pub use tensor::*;

pub use burn_common::reader::Reader; // Useful so that backends don't have to add `burn_common` as
