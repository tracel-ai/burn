#[macro_use]
extern crate derive_new;

mod external_data;
mod graph_state;
pub mod ir;
pub mod node;
mod phases;
mod pipeline;
mod processor;
mod proto_conversion;
mod protos;
mod registry;
mod tensor_store;

// Public API - only expose essentials
pub use ir::*;
pub use node::*;
pub use pipeline::{Error, OnnxGraphBuilder};
