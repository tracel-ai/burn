//! Enum-based node representation for ONNX operations
//!
//! This module contains the new NodeEnum type which provides compile-time type safety
//! for ONNX operations by encoding the operation type and its configuration in enum variants.

use super::argument::Argument;

/// Enum-based node representation
///
/// Each ONNX operation is represented as a separate enum variant containing
/// the operation-specific configuration. This provides:
/// - Compile-time type safety (no downcasting)
/// - No heap allocation for configs
/// - No vtable indirection
/// - Exhaustive matching enforced by compiler
#[derive(Debug, Clone)]
pub enum NodeEnum {
    /// Element-wise addition (no configuration needed)
    Add {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },

    /// Rectified Linear Unit activation (no configuration needed)
    Relu {
        name: String,
        inputs: Vec<Argument>,
        outputs: Vec<Argument>,
    },
}

impl NodeEnum {
    /// Get the node's name
    pub fn name(&self) -> &str {
        match self {
            NodeEnum::Add { name, .. } => name,
            NodeEnum::Relu { name, .. } => name,
        }
    }

    /// Get the node's inputs
    pub fn inputs(&self) -> &[Argument] {
        match self {
            NodeEnum::Add { inputs, .. } => inputs,
            NodeEnum::Relu { inputs, .. } => inputs,
        }
    }

    /// Get the node's outputs
    pub fn outputs(&self) -> &[Argument] {
        match self {
            NodeEnum::Add { outputs, .. } => outputs,
            NodeEnum::Relu { outputs, .. } => outputs,
        }
    }

    /// Get mutable access to the node's outputs
    pub fn outputs_mut(&mut self) -> &mut Vec<Argument> {
        match self {
            NodeEnum::Add { outputs, .. } => outputs,
            NodeEnum::Relu { outputs, .. } => outputs,
        }
    }

    /// Get the node type as a string (for debugging/logging)
    pub fn node_type_str(&self) -> &'static str {
        match self {
            NodeEnum::Add { .. } => "Add",
            NodeEnum::Relu { .. } => "Relu",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, DType, TensorType};

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
    fn test_add_node_creation() {
        let node = NodeEnum::Add {
            name: "test_add".to_string(),
            inputs: vec![
                create_test_argument("input1", 2),
                create_test_argument("input2", 2),
            ],
            outputs: vec![create_test_argument("output", 2)],
        };

        assert_eq!(node.name(), "test_add");
        assert_eq!(node.inputs().len(), 2);
        assert_eq!(node.outputs().len(), 1);
        assert_eq!(node.node_type_str(), "Add");
    }

    #[test]
    fn test_relu_node_creation() {
        let node = NodeEnum::Relu {
            name: "test_relu".to_string(),
            inputs: vec![create_test_argument("input", 3)],
            outputs: vec![create_test_argument("output", 3)],
        };

        assert_eq!(node.name(), "test_relu");
        assert_eq!(node.inputs().len(), 1);
        assert_eq!(node.outputs().len(), 1);
        assert_eq!(node.node_type_str(), "Relu");
    }

    #[test]
    fn test_outputs_mut() {
        let mut node = NodeEnum::Add {
            name: "test".to_string(),
            inputs: vec![],
            outputs: vec![create_test_argument("output", 2)],
        };

        // Modify output type
        node.outputs_mut()[0].ty = ArgType::Tensor(TensorType {
            dtype: DType::I64,
            rank: 3,
            static_shape: None,
        });

        match node.outputs()[0].ty {
            ArgType::Tensor(ref t) => {
                assert_eq!(t.dtype, DType::I64);
                assert_eq!(t.rank, 3);
            }
            _ => panic!("Expected tensor type"),
        }
    }
}
