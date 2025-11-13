//! # Loop
//!
//! Generic looping construct - executes loop body graph for a specified number of iterations.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Loop.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version
//! - **Opset 11**: Adds support for sequence types
//! - **Opset 13**: Clarified scoping rules
//! - **Opset 16**: Further refinements

use std::any::Any;

use crate::ir::{ArgType, DType, Node, NodeBuilder, NodeConfig, OnnxGraph};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

/// Helper function to transform type for scan output concatenation
/// Per ONNX Loop spec, scan outputs are concatenated along axis 0:
/// - Scalars (rank 0): unsqueezed to [1], then concatenated → [N, 1] (rank 2)
/// - Tensors (rank K): concatenated along axis 0 → [N*D0, D1, ...] (rank K, same rank)
fn add_concat_dimension(ty: ArgType) -> ArgType {
    use crate::ir::TensorType;

    match ty {
        // Scalar (rank 0) → unsqueeze to [1] → concat → [N, 1] (rank 2)
        ArgType::Scalar(dtype) => ArgType::Tensor(TensorType {
            dtype,
            rank: 2, // ONNX unsqueezes scalars before concat, resulting in rank 2
            static_shape: None,
        }),
        // Tensors: concatenated along axis 0 (same rank, first dim changes)
        ArgType::Tensor(mut tensor_type) => {
            // Clear static shape since num_iterations affects first dimension
            tensor_type.static_shape = None;
            ArgType::Tensor(tensor_type)
        }
        ArgType::Shape(_) => {
            // Shapes become rank-1 tensors when concatenated
            ArgType::Tensor(TensorType {
                dtype: DType::I64,
                rank: 1,
                static_shape: None,
            })
        }
    }
}

/// Configuration for Loop operation
#[derive(Debug, Clone)]
pub struct LoopConfig {
    pub body: OnnxGraph,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            body: OnnxGraph {
                nodes: vec![],
                inputs: vec![],
                outputs: vec![],
                _graph_data: None,
            },
        }
    }
}

impl NodeConfig for LoopConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

/// Loop node processor
pub struct LoopProcessor;

impl NodeProcessor for LoopProcessor {
    type Config = LoopConfig;

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 1)?;

        // Loop has at least 2 inputs: M (optional), cond (optional), v_initial... (variadic)
        // But M and cond can be empty strings in ONNX, so we need at least 2 inputs
        if node.inputs.len() < 2 {
            return Err(ProcessError::Custom(format!(
                "Loop node requires at least 2 inputs (M, cond), got {}",
                node.inputs.len()
            )));
        }

        // Validate M input (max trip count) - should be scalar int64 or empty
        if !node.inputs[0].name.is_empty() {
            let m_type = &node.inputs[0].ty;
            if !m_type.is_scalar() {
                return Err(ProcessError::TypeMismatch {
                    expected: "Scalar I64 (rank-0 tensor or Scalar type) or empty".to_string(),
                    actual: format!("{:?}", m_type),
                });
            }
            match m_type {
                ArgType::Scalar(dtype) if *dtype == DType::I64 => {
                    // Valid scalar int64
                }
                _ => {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Scalar I64 or empty".to_string(),
                        actual: format!("{:?}", m_type),
                    });
                }
            }
        }

        // Validate cond input - should be scalar bool or empty
        if !node.inputs[1].name.is_empty() {
            let cond_type = &node.inputs[1].ty;
            if !cond_type.is_scalar() {
                return Err(ProcessError::TypeMismatch {
                    expected: "Scalar Bool (rank-0 tensor or Scalar type) or empty".to_string(),
                    actual: format!("{:?}", cond_type),
                });
            }
            match cond_type {
                ArgType::Scalar(dtype) if *dtype == DType::Bool => {
                    // Valid scalar bool
                }
                _ => {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Scalar Bool or empty".to_string(),
                        actual: format!("{:?}", cond_type),
                    });
                }
            }
        }

        // Get body graph from config
        let config = self
            .extract_config(node, opset)
            .expect("Config extraction failed");
        let body_inputs = config.body.inputs.clone();
        let body_outputs = config.body.outputs.clone();

        // Body must have at least 2 inputs (iter_num, cond_in)
        if body_inputs.len() < 2 {
            return Err(ProcessError::Custom(format!(
                "Loop body must have at least 2 inputs (iter_num, cond_in), got {}",
                body_inputs.len()
            )));
        }

        // Second body input must be scalar bool (cond_in)
        let cond_in_type = &body_inputs[1].ty;
        if !cond_in_type.is_scalar() {
            return Err(ProcessError::TypeMismatch {
                expected: "Loop body second input (cond_in) must be Scalar Bool (rank-0 tensor or Scalar type)".to_string(),
                actual: format!("{:?}", cond_in_type),
            });
        }
        match cond_in_type {
            ArgType::Scalar(dtype) if dtype == &DType::Bool => {
                // Valid
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Loop body second input (cond_in) must be Scalar Bool".to_string(),
                    actual: format!("{:?}", cond_in_type),
                });
            }
        }

        // Body must have at least 1 output (cond_out)
        if body_outputs.is_empty() {
            return Err(ProcessError::Custom(
                "Loop body must have at least 1 output (cond_out)".to_string(),
            ));
        }

        // First body output must be scalar bool (cond_out)
        let cond_out_type = &body_outputs[0].ty;
        if !cond_out_type.is_scalar() {
            return Err(ProcessError::TypeMismatch {
                expected: "Loop body first output (cond_out) must be Scalar Bool (rank-0 tensor or Scalar type)".to_string(),
                actual: format!("{:?}", cond_out_type),
            });
        }
        match cond_out_type {
            ArgType::Scalar(dtype) if dtype == &DType::Bool => {
                // Valid
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Loop body first output (cond_out) must be Scalar Bool".to_string(),
                    actual: format!("{:?}", cond_out_type),
                });
            }
        }

        // Number of loop-carried dependencies in the body (excluding iter_num and cond_in)
        let num_body_loop_inputs = body_inputs.len() - 2;

        // Number of loop-carried outputs from body (excluding cond_out)
        // Body outputs = [cond_out, v_out..., scan_outputs...]
        let num_body_loop_outputs = body_outputs.len() - 1;

        // The body must output at least as many values as it takes as loop-carried inputs
        // (it can output more due to scan outputs)
        if num_body_loop_outputs < num_body_loop_inputs {
            return Err(ProcessError::Custom(format!(
                "Loop body must have at least {} loop-carried outputs to match {} loop-carried inputs (got {})",
                num_body_loop_inputs, num_body_loop_inputs, num_body_loop_outputs
            )));
        }

        // Create outputs based on body outputs (excluding cond_out)
        // Per ONNX spec:
        // - Loop-carried dependencies: final value matches body output type
        // - Scan outputs: concatenated along axis 0, adding a new dimension
        //
        // Body outputs: [cond_out, v_out_1, ..., v_out_N, scan_out_1, ..., scan_out_K]
        // Loop outputs: [v_final_1, ..., v_final_N, scan_1, ..., scan_K]
        let num_loop_carried_outputs = num_body_loop_inputs;

        if node.outputs.is_empty() {
            for (i, body_output) in body_outputs.iter().skip(1).enumerate() {
                let mut output = body_output.clone();

                // Scan outputs get concatenated, adding a dimension at axis 0
                if i >= num_loop_carried_outputs {
                    output.ty = add_concat_dimension(output.ty);
                }

                node.outputs.push(output);
            }
        } else {
            // Update types for existing outputs
            for (i, body_output) in body_outputs.iter().skip(1).enumerate() {
                if i < node.outputs.len() {
                    let mut ty = body_output.ty.clone();

                    // Scan outputs get concatenated, adding a dimension at axis 0
                    if i >= num_loop_carried_outputs {
                        ty = add_concat_dimension(ty);
                    }

                    node.outputs[i].ty = ty;
                }
            }
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        // Extract body graph from attributes
        let body = node
            .attrs
            .get("body")
            .ok_or_else(|| ProcessError::MissingAttribute("body".to_string()))?
            .clone()
            .into_graph();

        Ok(LoopConfig { body })
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Loop {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::AttributeValue;
    use crate::ir::{Argument, NodeType, OnnxGraph, TensorType};
    use crate::node::test_utils::TestNodeBuilder;
    use std::collections::HashMap;

    fn create_test_body(_num_loop_vars: usize) -> OnnxGraph {
        OnnxGraph {
            nodes: vec![],
            inputs: vec![
                // iter_num
                Argument {
                    name: "iter".to_string(),
                    ty: ArgType::Scalar(DType::I64),
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
                // cond_in
                Argument {
                    name: "cond_in".to_string(),
                    ty: ArgType::Scalar(DType::Bool),
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
                // v_in (loop-carried variable)
                Argument {
                    name: "v_in".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        dtype: DType::F32,
                        rank: 2,
                        static_shape: None,
                    }),
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
            ],
            outputs: vec![
                // cond_out
                Argument {
                    name: "cond_out".to_string(),
                    ty: ArgType::Scalar(DType::Bool),
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
                // v_out (loop-carried variable output)
                Argument {
                    name: "v_out".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        dtype: DType::F32,
                        rank: 2,
                        static_shape: None,
                    }),
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
            ],
            _graph_data: None,
        }
    }

    #[test]
    fn test_loop_basic() {
        let mut attrs = HashMap::new();
        attrs.insert(
            "body".to_string(),
            AttributeValue::Graph(create_test_body(1)),
        );

        let mut node = TestNodeBuilder::new(NodeType::Loop, "test_loop")
            .input_scalar("M", DType::I64)
            .input_scalar("cond", DType::Bool)
            .input_tensor_f32("v_initial", 2, None)
            .build();
        node.attrs = attrs;

        let processor = LoopProcessor;

        // Extract config first
        let _config = processor.extract_config(&node, 16).unwrap();

        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Loop should have 1 output (v_final)
        assert_eq!(node.outputs.len(), 1);
    }

    #[test]
    fn test_loop_invalid_trip_count() {
        let mut attrs = HashMap::new();
        attrs.insert(
            "body".to_string(),
            AttributeValue::Graph(create_test_body(1)),
        );

        let mut node = TestNodeBuilder::new(NodeType::Loop, "test_loop")
            .input_tensor_f32("M", 1, None) // Should be scalar, not tensor
            .input_scalar("cond", DType::Bool)
            .input_tensor_f32("v_initial", 2, None)
            .build();
        node.attrs = attrs;

        let processor = LoopProcessor;

        let _config = processor.extract_config(&node, 16).unwrap();

        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }
}
