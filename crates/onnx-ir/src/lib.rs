#[macro_use]
extern crate derive_new;

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
mod util;

// Public API - only expose essentials
pub use ir::*;
pub use node::*;
pub use pipeline::{OnnxIrError, parse_onnx, parse_onnx_bytes, parse_onnx_from_bytes};
