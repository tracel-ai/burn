//! # Loop
//!
//! Generic looping construct - executes loop body graph for a specified number of iterations.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Loop.html>
//!
//! ## Attributes
//! - `body` (graph): The graph to execute in each iteration
//!
//! ## Inputs
//! - `M` (optional, I): Maximum trip count (int64 scalar, can be empty for infinite loop with condition)
//! - `cond` (optional, B): Initial condition (bool scalar, empty means true)
//! - `v_initial` (variadic, V): Initial values of loop-carried dependencies
//!
//! ## Outputs
//! - `v_final_and_scan_outputs` (variadic, V): Final values of loop-carried dependencies and scan outputs
//!
//! ## Loop Body Inputs
//! - `iter_num` (I): Current iteration number (starts at 0)
//! - `cond_in` (B): Condition from previous iteration (or initial condition for iteration 0)
//! - `v_in` (variadic, V): Current values of loop-carried dependencies
//!
//! ## Loop Body Outputs
//! - `cond_out` (B): Condition for next iteration
//! - `v_out` (variadic, V): Updated values of loop-carried dependencies
//! - `scan_outputs` (variadic, V): Optional scan outputs accumulated across iterations
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version
//! - **Opset 11**: Adds support for sequence types
//! - **Opset 13**: Clarified scoping rules
//! - **Opset 16**: Further refinements

use std::any::Any;

use crate::ir::{ArgType, DType, Node, NodeConfig, OnnxGraph};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

/// Configuration for Loop operation
#[derive(Debug, Clone)]
pub struct LoopConfig {
    pub body: OnnxGraph,
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
    fn infer_types(
        &self,
        node: &mut Node,
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
        let config = node.config::<LoopConfig>();
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
            ArgType::Scalar(dtype) if *dtype == DType::Bool => {
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
            ArgType::Scalar(dtype) if *dtype == DType::Bool => {
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
        // Final values of loop-carried dependencies
        if node.outputs.is_empty() {
            for body_output in body_outputs.iter().skip(1) {
                node.outputs.push(body_output.clone());
            }
        } else {
            // Update types for existing outputs
            for (i, body_output) in body_outputs.iter().skip(1).enumerate() {
                if i < node.outputs.len() {
                    node.outputs[i].ty = body_output.ty.clone();
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
        // Extract body graph from attributes
        let body = node
            .attrs
            .get("body")
            .ok_or_else(|| ProcessError::MissingAttribute("body".to_string()))?
            .clone()
            .into_graph();

        Ok(Some(Box::new(LoopConfig { body })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::AttributeValue;
    use crate::ir::{Argument, NodeType, OnnxGraph, TensorType};
    use crate::node::test_utils::NodeBuilder;
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

        let mut node = NodeBuilder::new(NodeType::Loop, "test_loop")
            .input_scalar("M", DType::I64)
            .input_scalar("cond", DType::Bool)
            .input_tensor_f32("v_initial", 2, None)
            .build();
        node.attrs = attrs;

        let processor = LoopProcessor;

        // Extract config first
        let config = processor.extract_config(&node, 16).unwrap().unwrap();
        node.config = Some(config);

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

        let mut node = NodeBuilder::new(NodeType::Loop, "test_loop")
            .input_tensor_f32("M", 1, None) // Should be scalar, not tensor
            .input_scalar("cond", DType::Bool)
            .input_tensor_f32("v_initial", 2, None)
            .build();
        node.attrs = attrs;

        let processor = LoopProcessor;

        let config = processor.extract_config(&node, 16).unwrap().unwrap();
        node.config = Some(config);

        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }
}
