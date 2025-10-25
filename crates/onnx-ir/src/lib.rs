#[macro_use]
extern crate derive_new;

mod coalesce;
mod from_onnx;
pub mod ir;
pub mod node;
mod node_remap;
mod proto_conversion;
mod protos;
mod rank_inference;
pub mod util;

pub use from_onnx::convert_constant_value;
pub use from_onnx::element_type_from_proto;
pub use from_onnx::parse_onnx;
pub use ir::*;
