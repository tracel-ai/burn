//! Node processor trait and infrastructure for node-centric processing.
//!
//! This module defines the `NodeProcessor` trait with support for type preferences
//! and proper error handling.

use crate::ir::{Node, NodeConfig};
use std::collections::HashMap;

// Re-export registry types for backward compatibility
pub use crate::registry::{ProcessorRegistry, get_processor_registry};

/// Type preferences for node inputs
#[derive(Debug, Default, Clone)]
pub struct InputPreferences {
    preferences: HashMap<String, Vec<ArgPreference>>,
}

#[derive(Debug, Clone)]
pub enum ArgPreference {
    Scalar,
    Shape,
    Tensor,
}

impl InputPreferences {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(mut self, input_name: impl Into<String>, ty: ArgPreference) -> Self {
        self.preferences
            .entry(input_name.into())
            .or_default()
            .push(ty);
        self
    }

    pub fn get(&self, input_name: &str) -> &[ArgPreference] {
        self.preferences
            .get(input_name)
            .map_or(&[], |v| v.as_slice())
    }
}

/// Type preferences requested by consumers for a node's outputs
#[derive(Debug, Default, Clone)]
pub struct OutputPreferences {
    // output_name -> [(consumer_name, requested_type)]
    requests: HashMap<String, Vec<(String, ArgPreference)>>,
}

impl OutputPreferences {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(
        &mut self,
        output_name: impl Into<String>,
        consumer: impl Into<String>,
        ty: ArgPreference,
    ) {
        self.requests
            .entry(output_name.into())
            .or_default()
            .push((consumer.into(), ty));
    }

    pub fn get(&self, output_name: &str) -> &[(String, ArgPreference)] {
        self.requests.get(output_name).map_or(&[], |v| v.as_slice())
    }
}

/// Errors that can occur during node processing
#[derive(Debug, Clone)]
pub enum ProcessError {
    UnsupportedOpset {
        required: usize,
        actual: usize,
    },
    MissingInput(String),
    MissingOutput(String),
    InvalidInputCount {
        expected: usize,
        actual: usize,
    },
    InvalidOutputCount {
        expected: usize,
        actual: usize,
    },
    TypeMismatch {
        expected: String,
        actual: String,
    },
    ConflictingPreferences {
        output: String,
        details: Vec<String>,
    },
    MissingAttribute(String),
    InvalidAttribute {
        name: String,
        reason: String,
    },
    Custom(String),
}

/// Node-specific processing logic for type inference and configuration extraction
pub trait NodeProcessor: Send + Sync {
    /// Declare preferred types for inputs (propagated to producers as `output_preferences`)
    ///
    /// Preferences are requests, not requirements. Producers may honor them (e.g., Constant→Shape).
    fn input_preferences(
        &self,
        _node: &Node,
        _opset: usize,
    ) -> Result<Option<InputPreferences>, ProcessError> {
        Ok(None)
    }

    /// Convert constant inputs to static values (embedded in config, unreferenced constants removed later)
    fn lift_constants(&self, _node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        Ok(())
    }

    /// Infer output types given preferences from consumers
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError>;

    /// Extract config for codegen
    fn extract_config(
        &self,
        _node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        Ok(None)
    }
}

/// Default processor for nodes without specific implementations.
///
/// This processor passes through the input type to output.
pub(crate) struct DefaultProcessor;

impl NodeProcessor for DefaultProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Default: preserve input type
        crate::util::same_as_input(node);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, Node, NodeType, TensorType};

    struct TestProcessor;

    impl NodeProcessor for TestProcessor {
        fn infer_types(
            &self,
            node: &mut Node,
            _opset: usize,
            _output_preferences: &OutputPreferences,
        ) -> Result<(), ProcessError> {
            // Simple test: copy input type to output
            if !node.inputs.is_empty() && !node.outputs.is_empty() {
                node.outputs[0].ty = node.inputs[0].ty.clone();
            }
            Ok(())
        }
    }

    #[test]
    fn test_infer_outputs() {
        let processor = TestProcessor;
        let prefs = OutputPreferences::new();
        let mut node = Node {
            node_type: NodeType::Add,
            name: "test_node".to_string(),
            inputs: vec![Argument {
                name: "input".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 2,
                    static_shape: None,
                }),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::default(),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should match input type
        assert_eq!(node.outputs[0].ty, node.inputs[0].ty);
    }

    #[test]
    fn test_processor_registry() {
        let mut registry = ProcessorRegistry::new();

        // Register a processor
        registry.register(NodeType::Add, Box::new(TestProcessor));

        // Check if processor is registered
        assert!(registry.has_processor(&NodeType::Add));
        assert!(!registry.has_processor(&NodeType::Sub));
    }

    #[test]
    fn test_default_processor() {
        let processor = DefaultProcessor;
        let prefs = OutputPreferences::new();
        let mut node = Node {
            node_type: NodeType::Relu,
            name: "test_relu".to_string(),
            inputs: vec![Argument {
                name: "input".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 3,
                    static_shape: None,
                }),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::default(),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Default processor should preserve input type
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 3);
                assert_eq!(t.elem_type, ElementType::Float32);
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
