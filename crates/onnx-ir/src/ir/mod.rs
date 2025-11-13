//! ONNX Intermediate Representation (IR) types
//!
//! This module contains all the core types for representing ONNX models
//! in an intermediate representation suitable for code generation and analysis.

// Module declarations
pub mod argument;
pub mod attribute;
pub mod graph;
pub mod node;
pub mod node_enum;
pub mod node_type;
pub mod tensor_data_ext;

// Re-export burn-tensor's DType
pub use burn_tensor::DType;

// Re-exports from argument module
pub use argument::{ArgType, Argument, Rank, Shape, TensorId, TensorType, ValueSource};

// Re-exports from attribute module
pub use attribute::{AttributeValue, Attributes};

// Re-exports from graph module
pub use graph::OnnxGraph;

// Re-exports from node module
pub use node::{NodeBuilder, NodeConfig, RuntimeInputRef};

// Re-exports from node_enum module
pub use node_enum::Node;

// Re-exports from node_type module
pub use node_type::NodeType;

// Re-exports from tensor_data_ext module
pub use tensor_data_ext::{TensorData, TensorDataExt};
