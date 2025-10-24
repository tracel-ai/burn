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

pub use ir::*;
pub use pipeline::parse_onnx;
