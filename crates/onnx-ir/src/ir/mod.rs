//! ONNX Intermediate Representation (IR) types
//!
//! This module contains all the core types for representing ONNX models
//! in an intermediate representation suitable for code generation and analysis.

// Module declarations
pub mod argument;
pub mod attribute;
pub mod graph;
pub mod node;
pub mod tensor_data_ext;

// Re-export burn-tensor's DType
pub use burn_tensor::DType;

// Re-exports from argument module
pub use argument::{ArgType, Argument, DataId, Rank, Shape, TensorType, ValueSource};

// Re-exports from attribute module
pub use attribute::{AttributeValue, Attributes};

// Re-exports from graph module
pub use graph::{OnnxGraph, OnnxGraphBuilder};

// Re-exports from node module
pub use node::{Node, NodeBuilder, NodeConfig, NodeType, RuntimeInputRef};

// Re-exports from tensor_data_ext module
pub use tensor_data_ext::{TensorData, TensorDataExt};
