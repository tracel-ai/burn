//! Node processor trait and infrastructure for node-centric processing.
//!
//! This module defines the `NodeProcessor` trait which enables a demand-driven,
//! node-centric architecture where each node type declares its requirements
//! rather than having centralized orchestration code decide what to provide.

use crate::ir::{Node, NodeType};
use std::collections::HashMap;

/// Context provided to node processors during inference
#[derive(Debug, Clone)]
pub struct ProcessorContext {
    /// ONNX opset version being processed
    pub opset_version: i64,
}

impl ProcessorContext {
    /// Create a new processor context with the given opset version
    pub fn new(opset_version: i64) -> Self {
        Self { opset_version }
    }
}

/// Node-specific processing logic trait.
///
/// Each node type implements this trait to declare:
/// - Which opset versions it supports
/// - How to infer output types from input types
/// - Any special processing requirements
pub trait NodeProcessor {
    /// Get the supported opset version range for this node type.
    ///
    /// Returns (min_version, max_version) where:
    /// - min_version: minimum opset version required
    /// - max_version: maximum opset version supported (None = no upper limit)
    ///
    /// # Default
    ///
    /// The default implementation supports opset 16 and above.
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (16, None)
    }

    /// Infer output types from input types.
    ///
    /// This method should update the node's output arguments based on:
    /// - Input argument types
    /// - Node attributes
    /// - The provided context (including opset version)
    ///
    /// # Arguments
    ///
    /// * `node` - The node to process (mutable to update outputs)
    /// * `context` - Processing context with opset version and other metadata
    fn infer_outputs(&self, node: &mut Node, context: &ProcessorContext);
}

/// Registry for node processors.
///
/// This registry maps node types to their corresponding processor implementations.
/// It provides a centralized way to look up and use processors for different node types.
pub struct ProcessorRegistry {
    processors: HashMap<NodeType, Box<dyn NodeProcessor>>,
}

impl ProcessorRegistry {
    /// Create a new empty processor registry
    pub fn new() -> Self {
        Self {
            processors: HashMap::new(),
        }
    }

    /// Register a processor for a specific node type
    pub fn register(&mut self, node_type: NodeType, processor: Box<dyn NodeProcessor>) {
        self.processors.insert(node_type, processor);
    }

    /// Get the processor for a node type, or the default processor if not found
    pub fn get(&self, node_type: &NodeType) -> &dyn NodeProcessor {
        self.processors
            .get(node_type)
            .map(|b| b.as_ref())
            .unwrap_or(&DefaultProcessor)
    }

    /// Check if a processor is registered for a node type
    pub fn has_processor(&self, node_type: &NodeType) -> bool {
        self.processors.contains_key(node_type)
    }
}

impl Default for ProcessorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Default processor for nodes without specific implementations.
///
/// This processor uses the default opset range and passes through the input type to output.
struct DefaultProcessor;

impl NodeProcessor for DefaultProcessor {
    fn infer_outputs(&self, node: &mut Node, _context: &ProcessorContext) {
        // Default: preserve input type
        crate::util::same_as_input(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, Node, NodeType, TensorType};

    struct TestProcessor;

    impl NodeProcessor for TestProcessor {
        fn supported_opset_range(&self) -> (i64, Option<i64>) {
            (13, Some(18))
        }

        fn infer_outputs(&self, node: &mut Node, _context: &ProcessorContext) {
            // Simple test: copy input type to output
            if !node.inputs.is_empty() && !node.outputs.is_empty() {
                node.outputs[0].ty = node.inputs[0].ty.clone();
            }
        }
    }

    #[test]
    fn test_processor_context() {
        let ctx = ProcessorContext::new(16);
        assert_eq!(ctx.opset_version, 16);
    }

    #[test]
    fn test_default_opset_range() {
        let processor = TestProcessor;
        let (min, max) = processor.supported_opset_range();
        assert_eq!(min, 13);
        assert_eq!(max, Some(18));
    }

    #[test]
    fn test_infer_outputs() {
        let processor = TestProcessor;
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
                value: None,
                passed: true,
            }],
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::default(),
                value: None,
                passed: false,
            }],
            attrs: Default::default(),
        };

        let ctx = ProcessorContext::new(16);
        processor.infer_outputs(&mut node, &ctx);

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

        // Get registered processor
        let processor = registry.get(&NodeType::Add);
        let (min, max) = processor.supported_opset_range();
        assert_eq!(min, 13);
        assert_eq!(max, Some(18));

        // Get default processor for unregistered type
        let default_proc = registry.get(&NodeType::Sub);
        let (def_min, def_max) = default_proc.supported_opset_range();
        assert_eq!(def_min, 16);
        assert_eq!(def_max, None);
    }

    #[test]
    fn test_default_processor() {
        let processor = DefaultProcessor;
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
                value: None,
                passed: true,
            }],
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::default(),
                value: None,
                passed: false,
            }],
            attrs: Default::default(),
        };

        let ctx = ProcessorContext::new(16);
        processor.infer_outputs(&mut node, &ctx);

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
