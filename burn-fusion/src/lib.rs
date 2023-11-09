#![warn(missing_docs)]

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
pub(crate) use tensor::*;

pub use backend::*;
pub use fusion::*;
pub use handle::*;
