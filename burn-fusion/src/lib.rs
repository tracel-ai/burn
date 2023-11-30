#![warn(missing_docs)]

//! # Burn Fusion
//!
//! This library is a part of the Burn project. It is a standalone crate that
//! can be used to perform automatic operation fusion on backends that support it.

#[macro_use]
extern crate derive_new;

/// Client module exposing types to communicate with the fusion server.
pub mod client;
/// Graph module exposing all tensor operations that can be optimized.
pub mod graph;

mod backend;
mod fusion;
mod handle;
mod ops;
mod server;
mod tensor;

pub(crate) use server::*;

pub use backend::*;
pub use fusion::*;
pub use handle::*;
pub use tensor::*;
