#![warn(missing_docs)]

//! # Burn Autodiff
//!
//! This autodiff library is a part of the Burn project. It is a standalone crate
//! that can be used to perform automatic differentiation on tensors. It is
//! designed to be used with the Burn Tensor crate, but it can be used with any
//! tensor library that implements the `Backend` trait.

#[macro_use]
extern crate derive_new;

pub(crate) mod grads;
pub(crate) mod graph;
pub(crate) mod ops;
pub(crate) mod tensor;
pub(crate) mod utils;

mod backend;
pub use backend::*;

#[cfg(feature = "export_tests")]
mod tests;
