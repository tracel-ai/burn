//! ONNX node representation
//!
//! This module contains types for representing ONNX nodes, including their
//! configuration, inputs, outputs, and attributes.

use core::fmt;
use std::any::Any;

use super::argument::Argument;
use super::attribute::Attributes;
use super::node_type::NodeType;

/// Reference to a runtime input by name and index.
/// Used in configs to point to node inputs instead of storing stale copies.
#[derive(Debug, Clone, PartialEq)]
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
pub struct Node {
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

    /// Node-specific configuration (populated during processing)
    pub(crate) config: Option<Box<dyn NodeConfig>>,
}

impl Node {
    /// Get a reference to the node's attributes
    pub fn attrs(&self) -> &Attributes {
        &self.attrs
    }

    /// Get a reference to the node's configuration with automatic downcasting.
    /// Returns None if the config is not set or cannot be downcast to type T.
    ///
    /// # Example
    /// ```ignore
    /// if let Some(config) = node.get_config::<ArgMaxConfig>() {
    ///     // use config
    /// }
    /// ```
    pub fn get_config<T: NodeConfig + 'static>(&self) -> Option<&T> {
        self.config.as_ref()?.as_any().downcast_ref::<T>()
    }

    /// Get a reference to the node's configuration with automatic downcasting.
    /// Panics if the config is not set or cannot be downcast to type T.
    ///
    /// # Example
    /// ```ignore
    /// let config = node.config::<ArgMaxConfig>();
    /// // use config
    /// ```
    ///
    /// # Panics
    /// Panics if the config is not set or is the wrong type.
    pub fn config<T: NodeConfig + 'static>(&self) -> &T {
        self.get_config::<T>().unwrap_or_else(|| {
            panic!(
                "Node '{}' ({:?}) config is not set or has wrong type. Expected {}",
                self.name,
                self.node_type,
                std::any::type_name::<T>()
            )
        })
    }
}

// Custom Clone implementation since Box<dyn NodeConfig> doesn't auto-derive Clone
impl Clone for Node {
    fn clone(&self) -> Self {
        Self {
            node_type: self.node_type.clone(),
            name: self.name.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            attrs: self.attrs.clone(),
            config: self.config.as_ref().map(|c| c.clone_box()),
        }
    }
}

// Custom Debug implementation since Box<dyn NodeConfig> doesn't auto-derive Debug
impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("node_type", &self.node_type)
            .field("name", &self.name)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("attrs", &self.attrs)
            .field("config", &self.config.as_ref().map(|_| "Some(<config>)"))
            .finish()
    }
}
