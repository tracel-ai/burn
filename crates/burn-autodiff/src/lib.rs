#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

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
pub use graph::NodeId;
pub(crate) mod grad_sync;
pub(crate) mod tensor;
pub(crate) mod utils;

mod backend;

pub(crate) mod runtime;

pub use backend::*;
pub use grad_sync::api::start_gradient_sync_server;

/// A facade around for HashMap and HashSet.
/// This avoids elaborate import wrangling having to happen in every module.
mod collections {
    #[cfg(not(feature = "std"))]
    pub use hashbrown::{HashMap, HashSet};
    #[cfg(feature = "std")]
    pub use std::collections::{HashMap, HashSet};
}
