//! ONNX node representation
//!
//! This module contains types for representing ONNX nodes, including their
//! configuration, inputs, outputs, and attributes.

use std::any::Any;

use super::argument::Argument;
use super::attribute::Attributes;
use super::node_type::NodeType;

/// Reference to a runtime input by name and index.
/// Used in configs to point to node inputs instead of storing stale copies.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RuntimeInputRef {
    /// Name of the input argument
    pub name: String,
    /// Index in the node's inputs array
    pub input_index: usize,
}

impl RuntimeInputRef {
    pub fn new(name: String, input_index: usize) -> Self {
        Self { name, input_index }
    }
}

/// Trait for node-specific configuration
/// Each node type can have its own configuration struct that implements this trait
pub trait NodeConfig {
    /// Downcast to Any for type-safe retrieval
    fn as_any(&self) -> &dyn Any;

    /// Clone the config into a boxed trait object
    fn clone_box(&self) -> Box<dyn NodeConfig>;
}

/// Nodes produced by the ONNX parser
#[derive(Clone, Debug)]
pub struct NodeBuilder {
    /// The type of the node.
    /// This should be a valid ONNX operator.
    pub node_type: NodeType,

    /// The name of the node.
    pub name: String,

    /// The inputs of the node.
    pub inputs: Vec<Argument>,

    /// The outputs of the node.
    pub outputs: Vec<Argument>,

    /// ONNX attributes (opset-specific parameters)
    pub(crate) attrs: Attributes,
}

impl NodeBuilder {
    /// Create a new NodeBuilder
    pub fn new(
        node_type: NodeType,
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
        attrs: Attributes,
    ) -> Self {
        Self {
            node_type,
            name,
            inputs,
            outputs,
            attrs,
        }
    }

    /// Get a reference to the node's attributes
    pub fn attrs(&self) -> &Attributes {
        &self.attrs
    }
}
