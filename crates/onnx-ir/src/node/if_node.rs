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
    build_outer_scope_from_inputs,
};

/// Configuration for If operation
#[derive(Debug, Clone, new)]
pub struct IfConfig {
    pub then_branch: OnnxGraph,
    pub else_branch: OnnxGraph,
    /// Names of outer-scope references (in order corresponding to inputs[1..])
    /// These are the original sanitized ONNX names that subgraphs reference
    #[new(default)]
    pub scope_ref_names: Vec<String>,
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
        // Validate condition input is bool
        // Per ONNX spec, the condition must be a tensor containing a single element,
        // but we allow any bool tensor/scalar as some models may not be strictly conformant
        let condition = &node.inputs[0].ty;
        let is_bool = match condition {
            ArgType::Scalar(dtype) => dtype.is_bool(),
            ArgType::Tensor(tensor) => tensor.dtype.is_bool(),
            ArgType::Shape(_) => false,
        };

        if !is_bool {
            return Err(ProcessError::TypeMismatch {
                expected: "Bool tensor or scalar".to_string(),
                actual: format!("{:?}", condition),
            });
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
        // When branches have different types, merge them to find the common type.
        // For tensors with different ranks, use the lower rank (common base type).
        // ONLY update types, preserve existing output structure (names set by add_node)

        // If outputs don't exist yet, create them from branch outputs
        if node.outputs.is_empty() {
            for (then_output, else_output) in then_outputs.iter().zip(else_outputs.iter()) {
                let merged_ty = merge_branch_types(&then_output.ty, &else_output.ty);
                let mut output = then_output.clone();
                output.ty = merged_ty;
                node.outputs.push(output);
            }
        } else {
            // Update types for existing outputs (preserves names set by add_node)
            for (i, (then_output, else_output)) in
                then_outputs.iter().zip(else_outputs.iter()).enumerate()
            {
                let merged_ty = merge_branch_types(&then_output.ty, &else_output.ty);

                // Only update the type, keep the existing name
                if i < node.outputs.len() {
                    node.outputs[i].ty = merged_ty;
                }
            }
        }

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
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

        // Build outer scope types map from additional inputs (beyond ONNX inputs)
        // These are outer-scope references that were added during node conversion
        let outer_scope = build_outer_scope_from_inputs(node);

        // Handle DeferredGraph and Graph
        let then_branch = match then_attr {
            crate::ir::AttributeValue::DeferredGraph(deferred) => {
                // Build the subgraph now with outer-scope types
                log::debug!(
                    "Building deferred then_branch subgraph with {} outer-scope types",
                    outer_scope.len()
                );
                deferred
                    .build_graph_with_outer_scope(outer_scope.clone())
                    .map_err(|e| {
                        ProcessError::Custom(format!("Failed to build then_branch: {:?}", e))
                    })?
            }
            crate::ir::AttributeValue::Graph(g) => g,
            _ => {
                return Err(ProcessError::Custom(
                    "Expected DeferredGraph or Graph for then_branch".to_string(),
                ));
            }
        };

        let else_branch = match else_attr {
            crate::ir::AttributeValue::DeferredGraph(deferred) => {
                // Build the subgraph now with outer-scope types
                log::debug!(
                    "Building deferred else_branch subgraph with {} outer-scope types",
                    outer_scope.len()
                );
                deferred
                    .build_graph_with_outer_scope(outer_scope)
                    .map_err(|e| {
                        ProcessError::Custom(format!("Failed to build else_branch: {:?}", e))
                    })?
            }
            crate::ir::AttributeValue::Graph(g) => g,
            _ => {
                return Err(ProcessError::Custom(
                    "Expected DeferredGraph or Graph for else_branch".to_string(),
                ));
            }
        };

        // Get the scope ref names for use in code generation
        let scope_ref_names: Vec<String> = node
            .attrs
            .get("__scope_ref_names")
            .and_then(|v| match v {
                crate::ir::AttributeValue::Strings(names) => Some(names.clone()),
                _ => None,
            })
            .unwrap_or_default();

        Ok(IfConfig {
            then_branch,
            else_branch,
            scope_ref_names,
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

/// Merge branch output types when they differ.
///
/// ONNX allows If branches to have outputs with different shapes/ranks as long as
/// they are "compatible" at runtime. For example, one branch might output `[1, N, 128]`
/// while another outputs `[N, 128]`. At runtime, only one branch executes.
///
/// For static type inference, we need to choose a consistent type. This function:
/// 1. Returns the type if both branches match exactly
/// 2. For tensors with same dtype but different ranks, uses the higher rank
/// 3. For incompatible dtypes, logs a warning and uses then_branch
///
/// We prefer higher rank because lower-rank outputs are often the result of
/// squeeze operations that can be unsqueezed to match the higher rank.
fn merge_branch_types(then_ty: &ArgType, else_ty: &ArgType) -> ArgType {
    use crate::ir::TensorType;

    if then_ty == else_ty {
        return then_ty.clone();
    }

    match (then_ty, else_ty) {
        (ArgType::Tensor(then_tensor), ArgType::Tensor(else_tensor)) => {
            // Validate dtype compatibility
            if then_tensor.dtype != else_tensor.dtype {
                log::warn!(
                    "If branches have incompatible dtypes: then={:?}, else={:?}. Using then_branch dtype.",
                    then_tensor.dtype,
                    else_tensor.dtype
                );
                return then_ty.clone();
            }

            // Same dtype, different ranks - use higher rank
            if then_tensor.rank != else_tensor.rank {
                let chosen_rank = std::cmp::max(then_tensor.rank, else_tensor.rank);
                log::debug!(
                    "If branches have different ranks: then={}, else={}. Using rank {}.",
                    then_tensor.rank,
                    else_tensor.rank,
                    chosen_rank
                );
                return ArgType::Tensor(TensorType {
                    dtype: then_tensor.dtype,
                    rank: chosen_rank,
                    static_shape: None, // Can't determine static shape when ranks differ
                });
            }

            // Same dtype and rank but different static shapes - clear static shape
            log::debug!("If branches have different static shapes. Using dynamic shape.");
            ArgType::Tensor(TensorType {
                dtype: then_tensor.dtype,
                rank: then_tensor.rank,
                static_shape: None,
            })
        }
        (ArgType::Scalar(then_dtype), ArgType::Scalar(else_dtype)) => {
            if then_dtype != else_dtype {
                log::warn!(
                    "If branches have incompatible scalar dtypes: then={:?}, else={:?}. Using then_branch.",
                    then_dtype,
                    else_dtype
                );
            }
            then_ty.clone()
        }
        _ => {
            // Different type categories (Tensor vs Scalar vs Shape)
            log::warn!(
                "If branches have incompatible type categories: then={:?}, else={:?}. Using then_branch.",
                then_ty,
                else_ty
            );
            then_ty.clone()
        }
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
