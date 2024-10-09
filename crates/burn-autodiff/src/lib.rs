#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! # Burn Autodiff
//!
//! This autodiff library is a part of the Burn project. It is a standalone crate
//! that can be used to perform automatic differentiation on tensors. It is
//! designed to be used with the Burn Tensor crate, but it can be used with any
//! tensor library that implements the `Backend` trait.

#[macro_use]
extern crate derive_new;

extern crate alloc;

/// Checkpoint module.
pub mod checkpoint;
/// Gradients module.
pub mod grads;
/// Operation module.
pub mod ops;

pub(crate) mod graph;
// Exported for backend extension
pub use graph::NodeID;
pub(crate) mod tensor;
pub(crate) mod utils;

mod backend;
mod bridge;

pub(crate) mod runtime;

pub use backend::*;
pub use bridge::*;

#[cfg(feature = "export_tests")]
mod tests;
