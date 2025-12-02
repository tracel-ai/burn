//! # If
//!
//! Conditional execution - executes either then_branch or else_branch based on a boolean condition.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__If.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version
//! - **Opset 11**: Adds support for sequence types
//! - **Opset 13**: Clarified scoping rules
//! - **Opset 16**: Further refinements

use derive_new::new;
use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, OnnxGraph, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Configuration for If operation
#[derive(Debug, Clone, new)]
pub struct IfConfig {
    pub then_branch: OnnxGraph,
    pub else_branch: OnnxGraph,
}

/// Node representation for If operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct IfNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: IfConfig,
}

/// If node processor
pub(crate) struct IfProcessor;

impl NodeProcessor for IfProcessor {
    type Config = IfConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Range(1, 2147483647),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate condition input is scalar bool
        let condition = &node.inputs[0].ty;
        if !condition.is_scalar() {
            return Err(ProcessError::TypeMismatch {
                expected: "Scalar Bool (rank-0 tensor or Scalar type)".to_string(),
                actual: format!("{:?}", condition),
            });
        }

        match condition {
            ArgType::Scalar(dtype) if dtype.is_bool() => {
                // Valid scalar bool condition
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Scalar Bool".to_string(),
                    actual: format!("{:?}", condition),
                });
            }
        }

        // Get branches from config (clone to avoid borrow checker issues)
        let config = self
            .extract_config(node, opset)
            .expect("Config extraction failed");
        let then_outputs = config.then_branch.outputs.clone();
        let else_outputs = config.else_branch.outputs.clone();

        // Both branches must produce the same number of outputs
        if then_outputs.len() != else_outputs.len() {
            return Err(ProcessError::Custom(format!(
                "If node branches must have same number of outputs: then={}, else={}",
                then_outputs.len(),
                else_outputs.len()
            )));
        }

        // Infer output types from branches
        // Both branches should have the same output types, but we'll take then_branch as canonical
        // ONLY update types, preserve existing output structure (names set by add_node)

        // If outputs don't exist yet, create them from branch outputs
        if node.outputs.is_empty() {
            for then_output in then_outputs.iter() {
                node.outputs.push(then_output.clone());
            }
        } else {
            // Update types for existing outputs (preserves names set by add_node)
            for (i, then_output) in then_outputs.iter().enumerate() {
                let else_output = &else_outputs[i];

                // Validate that output types are compatible
                if then_output.ty != else_output.ty {
                    log::warn!(
                        "If node output {} types differ between branches: then={:?}, else={:?}",
                        i,
                        then_output.ty,
                        else_output.ty
                    );
                }

                // Only update the type, keep the existing name
                if i < node.outputs.len() {
                    node.outputs[i].ty = then_output.ty.clone();
                }
            }
        }

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, opset: usize) -> Result<Self::Config, ProcessError> {
        // Extract then_branch and else_branch from attributes
        let then_attr = node
            .attrs
            .get("then_branch")
            .ok_or_else(|| ProcessError::MissingAttribute("then_branch".to_string()))?
            .clone();

        let else_attr = node
            .attrs
            .get("else_branch")
            .ok_or_else(|| ProcessError::MissingAttribute("else_branch".to_string()))?
            .clone();

        // Handle both Graph and GraphBuilder
        let then_branch = match then_attr {
            crate::ir::AttributeValue::Graph(g) => g,
            crate::ir::AttributeValue::GraphBuilder(mut builder) => {
                // Convert NodeBuilders to Nodes
                let nodes = crate::ir::graph::finalize_graph_nodes(&mut builder.nodes, opset);
                let value_store = builder
                    .graph_state
                    .as_ref()
                    .map(|gs| gs.borrow().build_value_store());
                crate::ir::OnnxGraph {
                    nodes,
                    inputs: std::mem::take(&mut builder.inputs),
                    outputs: std::mem::take(&mut builder.outputs),
                    value_store,
                }
            }
            _ => {
                return Err(ProcessError::Custom(
                    "Expected Graph or GraphBuilder for then_branch".to_string(),
                ));
            }
        };

        let else_branch = match else_attr {
            crate::ir::AttributeValue::Graph(g) => g,
            crate::ir::AttributeValue::GraphBuilder(mut builder) => {
                // Convert NodeBuilders to Nodes
                let nodes = crate::ir::graph::finalize_graph_nodes(&mut builder.nodes, opset);
                let value_store = builder
                    .graph_state
                    .as_ref()
                    .map(|gs| gs.borrow().build_value_store());
                crate::ir::OnnxGraph {
                    nodes,
                    inputs: std::mem::take(&mut builder.inputs),
                    outputs: std::mem::take(&mut builder.outputs),
                    value_store,
                }
            }
            _ => {
                return Err(ProcessError::Custom(
                    "Expected Graph or GraphBuilder for else_branch".to_string(),
                ));
            }
        };

        Ok(IfConfig {
            then_branch,
            else_branch,
        })
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::If(IfNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    use crate::ir::AttributeValue;
    use crate::ir::{Argument, NodeType, OnnxGraph, TensorType};
    use crate::node::test_utils::TestNodeBuilder;
    use std::collections::HashMap;

    fn create_test_branch(output_rank: usize, dtype: DType) -> OnnxGraph {
        OnnxGraph {
            nodes: vec![],
            inputs: vec![],
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::Tensor(TensorType {
                    dtype,
                    rank: output_rank,
                    static_shape: None,
                }),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            value_store: None,
        }
    }

    #[test]
    fn test_if_basic() {
        let mut attrs = HashMap::new();
        attrs.insert(
            "then_branch".to_string(),
            AttributeValue::Graph(create_test_branch(2, DType::F32)),
        );
        attrs.insert(
            "else_branch".to_string(),
            AttributeValue::Graph(create_test_branch(2, DType::F32)),
        );

        let mut node = TestNodeBuilder::new(NodeType::If, "test_if")
            .input_scalar("cond", DType::Bool)
            .build();
        node.attrs = attrs;

        let processor = IfProcessor;

        // Extract config first
        let _config = processor.extract_config(&node, 16).unwrap();

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert_eq!(node.outputs.len(), 1);
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_if_invalid_condition() {
        let mut attrs = HashMap::new();
        attrs.insert(
            "then_branch".to_string(),
            AttributeValue::Graph(create_test_branch(2, DType::F32)),
        );
        attrs.insert(
            "else_branch".to_string(),
            AttributeValue::Graph(create_test_branch(2, DType::F32)),
        );

        let mut node = TestNodeBuilder::new(NodeType::If, "test_if")
            .input_tensor_f32("cond", 1, None) // Tensor instead of scalar
            .build();
        node.attrs = attrs;

        let processor = IfProcessor;

        // Extract config first
        let _config = processor.extract_config(&node, 16).unwrap();

        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_if_branch_output_count_mismatch() {
        let mut attrs = HashMap::new();
        // then_branch has 1 output
        attrs.insert(
            "then_branch".to_string(),
            AttributeValue::Graph(create_test_branch(2, DType::F32)),
        );

        // else_branch has 2 outputs
        let mut else_branch = create_test_branch(2, DType::F32);
        else_branch.outputs.push(Argument {
            name: "output2".to_string(),
            ty: ArgType::Tensor(TensorType {
                dtype: DType::F32,
                rank: 2,
                static_shape: None,
            }),
            value_source: crate::ir::ValueSource::Dynamic,
            value_store: None,
        });
        attrs.insert(
            "else_branch".to_string(),
            AttributeValue::Graph(else_branch),
        );

        let mut node = TestNodeBuilder::new(NodeType::If, "test_if")
            .input_scalar("cond", DType::Bool)
            .build();
        node.attrs = attrs;

        let processor = IfProcessor;

        // Extract config first
        let _config = processor.extract_config(&node, 16).unwrap();

        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }
}
