//! # If
//!
//! Conditional execution - executes either then_branch or else_branch based on a boolean condition.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__If.html>
//!
//! ## Attributes
//! - `then_branch` (graph): Graph to execute if condition is true
//! - `else_branch` (graph): Graph to execute if condition is false
//!
//! ## Inputs
//! - `cond` (B): Scalar boolean condition
//! - Additional inputs: Any inputs referenced by the subgraphs (implicit)
//!
//! ## Outputs
//! - Outputs from the executed branch (number and types match between branches)
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version
//! - **Opset 11**: Adds support for sequence types
//! - **Opset 13**: Clarified scoping rules
//! - **Opset 16**: Further refinements

use std::any::Any;

use crate::ir::{ArgType, DType, Node, NodeConfig, OnnxGraph};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

/// Configuration for If operation
#[derive(Debug, Clone)]
pub struct IfConfig {
    pub then_branch: OnnxGraph,
    pub else_branch: OnnxGraph,
}

impl NodeConfig for IfConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

/// If node processor
pub struct IfProcessor;

impl NodeProcessor for IfProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 1)?;
        crate::processor::validate_input_count(node, 1)?;

        // Validate condition input is scalar bool
        let condition = &node.inputs[0].ty;
        match condition {
            ArgType::Scalar(dtype) if *dtype == DType::Bool => {
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
        let config = node.config::<IfConfig>();
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

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Extract then_branch and else_branch from attributes
        let then_branch = node
            .attrs
            .get("then_branch")
            .ok_or_else(|| ProcessError::MissingAttribute("then_branch".to_string()))?
            .clone()
            .into_graph();

        let else_branch = node
            .attrs
            .get("else_branch")
            .ok_or_else(|| ProcessError::MissingAttribute("else_branch".to_string()))?
            .clone()
            .into_graph();

        Ok(Some(Box::new(IfConfig {
            then_branch,
            else_branch,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::AttributeValue;
    use crate::ir::{Argument, NodeType, OnnxGraph, TensorType};
    use crate::node::test_utils::NodeBuilder;
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
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            _graph_data: None,
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

        let mut node = NodeBuilder::new(NodeType::If, "test_if")
            .input_scalar("cond", DType::Bool)
            .build();
        node.attrs = attrs;

        let processor = IfProcessor;

        // Extract config first
        let config = processor.extract_config(&node, 16).unwrap().unwrap();
        node.config = Some(config);

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

        let mut node = NodeBuilder::new(NodeType::If, "test_if")
            .input_tensor_f32("cond", 1, None) // Tensor instead of scalar
            .build();
        node.attrs = attrs;

        let processor = IfProcessor;

        // Extract config first
        let config = processor.extract_config(&node, 16).unwrap().unwrap();
        node.config = Some(config);

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
            data_id: None,
            value_source: crate::ir::ValueSource::Dynamic,
            value_store: None,
        });
        attrs.insert(
            "else_branch".to_string(),
            AttributeValue::Graph(else_branch),
        );

        let mut node = NodeBuilder::new(NodeType::If, "test_if")
            .input_scalar("cond", DType::Bool)
            .build();
        node.attrs = attrs;

        let processor = IfProcessor;

        // Extract config first
        let config = processor.extract_config(&node, 16).unwrap().unwrap();
        node.config = Some(config);

        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }
}
