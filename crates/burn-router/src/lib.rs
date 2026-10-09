#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Records tensor operations as IR and forwards them to wherever they execute.
//!
//! [`BackendRouter`] is a backend whose operations are not run in place: each one is
//! described as `burn_ir::OperationIr` and sent through a [`RouterChannel`] to a client. On
//! the receiving side, a [`TensorInterpreter`] replays the operations on a real backend.
//!
//! This is the layer under Burn's remote execution (`burn-remote` sends the operations over
//! the network) and graph capture (`burn-capture` records them without executing). Custom
//! operations from backend extensions travel through a [`CustomOpRegistry`].
//!
//! Applications do not use this crate directly.
//!
//! # Feature flags
//!
//! - `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
//! - `fusion`: fuse routed operations before they are sent.
//! - `tracing`: instrument operations with the `tracing` crate.

mod backend;
mod bridge;
mod channel;
mod client;
mod custom_op;
#[cfg(feature = "fusion")]
mod fusion;
mod graph;
mod interpreter;
mod ops;
mod tensor;

pub use backend::*;
pub use bridge::*;
pub use channel::*;
pub use client::*;
pub use custom_op::*;
#[cfg(feature = "fusion")]
pub use fusion::*;
pub use graph::*;
pub use interpreter::*;
pub use tensor::*;

extern crate alloc;
