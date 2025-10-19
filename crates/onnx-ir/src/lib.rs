#[macro_use]
extern crate derive_new;

mod graph_state;
pub mod ir;
pub mod node;
mod phases;
mod pipeline;
pub mod processor;
mod proto_conversion;
mod protos;
mod registry;
mod tensor_store;
pub mod util;

pub use ir::*;
pub use pipeline::parse_onnx;
pub use proto_conversion::{convert_constant_value, element_type_from_proto};
pub use util::{validate_input_count, validate_min_inputs, validate_opset, validate_output_count};
