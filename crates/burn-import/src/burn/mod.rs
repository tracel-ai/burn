/// The graph module.
pub mod graph;

mod codegen;
mod node_codegen; // Implements NodeCodegen<PS> for onnx_ir::Node
mod node_traits;

mod imports;

mod argument_helpers;
mod scope;

pub(crate) use argument_helpers::*;
pub(crate) use codegen::ToTokens;
pub(crate) use imports::*;
pub(crate) use node_traits::{Field, TensorKind};
pub(crate) use scope::*;

pub(crate) mod node;
