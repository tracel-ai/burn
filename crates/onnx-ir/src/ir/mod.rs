//! ONNX Intermediate Representation (IR) types
//!
//! This module contains all the core types for representing ONNX models
//! in an intermediate representation suitable for code generation and analysis.

// Module declarations
mod argument;
mod attribute;
pub(crate) mod graph;
mod node;
mod tensor_data_ext;

pub(crate) use attribute::{AttributeValue, Attributes, DeferredGraph, OuterScopeTypes};
pub(crate) use graph::OnnxGraphBuilder;
pub use node::RuntimeInputRef;
pub(crate) use node::{NodeType, RawNode};
pub use tensor_data_ext::TensorDataExt;

// Re-exports
pub use argument::{ArgType, Argument, DataId, Rank, Shape, TensorType, ValueSource};
pub use burn_tensor::DType;
pub use graph::OnnxGraph;
pub use node::Node;
pub use tensor_data_ext::TensorData;
