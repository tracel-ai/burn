#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! # Burn Autodiff
//!
//! Reverse-mode automatic differentiation as a backend decorator.
//!
//! [`Autodiff`] wraps any backend `B` and records the operations needed to compute gradients.
//! Only first-order derivatives are supported.
//!
//! Most applications do not name this type. Enable the `autodiff` feature of `burn` (also
//! enabled by `train`) and turn autodiff on for a device:
//!
//! ```rust,ignore
//! let device = Device::wgpu(Default::default()).autodiff();
//! let x = Tensor::<2>::ones([2, 2], &device).require_grad();
//! let grads = (x.clone() * 3.0).sum().backward();
//! let x_grad = x.grad(&grads).unwrap();
//! ```
//!
//! Dispatch then routes operations on that device through [`Autodiff`]. Use this crate
//! directly when implementing a backend extension that needs custom backward passes.
//!
//! # Gradient checkpointing
//!
//! The second type parameter selects a [`CheckpointStrategy`](checkpoint::strategy::CheckpointStrategy).
//! [`NoCheckpointing`](checkpoint::strategy::NoCheckpointing) keeps every activation needed by
//! the backward pass. [`BalancedCheckpointing`](checkpoint::strategy::BalancedCheckpointing)
//! recomputes cheap operations instead of storing their outputs, trading compute for memory.
//! At the `Device` level, `device.autodiff().gradient_checkpointing()` selects it.
//!
//! # Feature flags
//!
//! - `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
//! - `tracing`: instrument operations with the `tracing` crate.

#[macro_use]
extern crate derive_new;

extern crate alloc;

/// Checkpoint module.
pub mod checkpoint;
#[cfg(feature = "std")]
/// Distributed utils.
pub mod distributed;
/// Gradients module.
pub mod grads;
/// Operation module.
pub mod ops;

pub(crate) mod graph;
// Exported for backend extension
pub use graph::NodeId;
pub(crate) mod tensor;
pub(crate) mod utils;

mod backend;

pub(crate) mod runtime;

pub use backend::*;

/// A facade around for HashMap and HashSet.
/// This avoids elaborate import wrangling having to happen in every module.
mod collections {
    #[cfg(not(feature = "std"))]
    pub use hashbrown::{HashMap, HashSet};
    #[cfg(feature = "std")]
    pub use std::collections::{HashMap, HashSet};
}
