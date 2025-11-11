//! NodeBuilder - Intermediate mutable struct for node processing
//!
//! This module contains the NodeBuilder type which holds temporary state
//! during the node processing pipeline. It contains both attributes (from ONNX)
//! and configuration (extracted and validated) that are used during processing.

use super::argument::Argument;
use super::attribute::Attributes;
use super::node::NodeConfig;
use super::node_type::NodeType;

/// Intermediate mutable struct used during processing phases
///
/// This struct holds all the state needed during node processing:
/// - `attrs`: Raw ONNX attributes (available throughout processing)
/// - `config`: Extracted and validated configuration (set by extract_config phase)
///
/// The processing flow is:
/// 1. Create NodeBuilder from protobuf (has attrs, no config)
/// 2. lift_constants() - may modify inputs
/// 3. extract_config() - sets builder.config from builder.attrs
/// 4. infer_types() - uses builder.config to determine output types
/// 5. build_node() - converts to NodeEnum (keeps config, discards attrs)
pub struct NodeBuilder {
    /// The type of the node (ONNX operator type)
    pub node_type: NodeType,

    /// The name of the node
    pub name: String,

    /// The inputs of the node
    pub inputs: Vec<Argument>,

    /// The outputs of the node
    pub outputs: Vec<Argument>,

    /// ONNX attributes (raw, from protobuf)
    /// Available throughout processing, discarded at the end
    pub attrs: Attributes,

    /// Node-specific configuration (extracted from attrs)
    /// Set by extract_config(), used by infer_types()
    pub config: Option<Box<dyn NodeConfig>>,
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
            config: None,
        }
    }

    /// Get a reference to the node's configuration with automatic downcasting.
    /// Returns None if the config is not set or cannot be downcast to type T.
    ///
    /// # Example
    /// ```ignore
    /// if let Some(config) = builder.get_config::<ArgMaxConfig>() {
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
    /// let config = builder.config::<ArgMaxConfig>();
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

    /// Get the node's attributes
    pub fn attrs(&self) -> &Attributes {
        &self.attrs
    }
}

// Custom Clone implementation since Box<dyn NodeConfig> doesn't auto-derive Clone
impl Clone for NodeBuilder {
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
impl std::fmt::Debug for NodeBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeBuilder")
            .field("node_type", &self.node_type)
            .field("name", &self.name)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("attrs", &self.attrs)
            .field("config", &self.config.as_ref().map(|_| "<config>"))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, DType, TensorType};
    use std::collections::HashMap;

    fn create_test_argument(name: &str, rank: usize) -> Argument {
        Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank,
                static_shape: None,
            }),
            value_source: crate::ir::ValueSource::Dynamic,
            value_store: None,
        }
    }

    #[test]
    fn test_builder_creation() {
        let builder = NodeBuilder::new(
            NodeType::Add,
            "test_add".to_string(),
            vec![
                create_test_argument("input1", 2),
                create_test_argument("input2", 2),
            ],
            vec![create_test_argument("output", 2)],
            HashMap::new(),
        );

        assert_eq!(builder.node_type, NodeType::Add);
        assert_eq!(builder.name, "test_add");
        assert_eq!(builder.inputs.len(), 2);
        assert_eq!(builder.outputs.len(), 1);
        assert!(builder.config.is_none());
    }

    #[test]
    fn test_builder_clone() {
        let builder = NodeBuilder::new(
            NodeType::Relu,
            "test_relu".to_string(),
            vec![create_test_argument("input", 3)],
            vec![create_test_argument("output", 3)],
            HashMap::new(),
        );

        let cloned = builder.clone();
        assert_eq!(cloned.name, builder.name);
        assert_eq!(cloned.node_type, builder.node_type);
        assert_eq!(cloned.inputs.len(), builder.inputs.len());
    }
}
