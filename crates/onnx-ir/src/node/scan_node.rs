//! # Scan
//!
//! Generic scan construct - iterates over input sequences while maintaining state variables.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Scan.html>
//!
//! ## Opset Versions
//! - **Opset 8**: Initial version
//! - **Opset 9**: Added scan_input_axes
//! - **Opset 11**: Clarified behavior
//! - **Opset 16**: Further refinements

use crate::ir::{ArgType, Node, NodeBuilder, OnnxGraph};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

/// Configuration for Scan operation
#[derive(Debug, Clone)]
pub struct ScanConfig {
    pub body: OnnxGraph,
    pub num_scan_inputs: i64,
    pub scan_input_directions: Vec<i64>,
    pub scan_output_directions: Vec<i64>,
    pub scan_input_axes: Vec<i64>,
    pub scan_output_axes: Vec<i64>,
}

/// Scan node processor
pub(crate) struct ScanProcessor;

impl NodeProcessor for ScanProcessor {
    type Config = ScanConfig;

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 8)?;

        // Get config to determine number of state variables and scan inputs
        let config = self
            .extract_config(node, opset)
            .expect("Config extraction failed");
        let num_scan_inputs = config.num_scan_inputs as usize;

        // Total inputs = num_state_vars + num_scan_inputs
        if node.inputs.len() < num_scan_inputs {
            return Err(ProcessError::Custom(format!(
                "Scan requires at least {} inputs (num_scan_inputs), got {}",
                num_scan_inputs,
                node.inputs.len()
            )));
        }

        let num_state_vars = node.inputs.len() - num_scan_inputs;

        // Get body outputs to determine output types
        let body_outputs = config.body.outputs.clone();

        // Body outputs = [state_vars_out..., scan_outputs...]
        // Minimum: num_state_vars (state variables must be preserved)
        if body_outputs.len() < num_state_vars {
            return Err(ProcessError::Custom(format!(
                "Scan body must have at least {} outputs (state variables), got {}",
                num_state_vars,
                body_outputs.len()
            )));
        }

        let _num_scan_outputs = body_outputs.len() - num_state_vars;

        // Scan node outputs = [final_state_vars..., scan_output_sequences...]
        // Final state vars have same type as body output state vars
        // Scan output sequences have rank increased by 1 for the sequence dimension

        if node.outputs.is_empty() {
            // Create outputs: first the final state variables
            for body_output in body_outputs.iter().take(num_state_vars) {
                node.outputs.push(body_output.clone());
            }

            // Then the scan output sequences
            // Scan outputs have rank increased by 1 compared to body outputs
            for body_scan_output in body_outputs.iter().skip(num_state_vars) {
                let mut output = body_scan_output.clone();
                output.name = format!("{}_sequence", body_scan_output.name);

                // Increase rank by 1 for sequence dimension
                match &body_scan_output.ty {
                    ArgType::Tensor(body_tensor) => {
                        output.ty = ArgType::Tensor(crate::ir::TensorType {
                            dtype: body_tensor.dtype,
                            rank: body_tensor.rank + 1,
                            static_shape: None, // Dynamic shape for now
                        });
                    }
                    _ => {
                        // Keep same type if not a tensor
                    }
                }

                node.outputs.push(output);
            }
        } else {
            // Update types for existing outputs
            for (output, body_output) in node
                .outputs
                .iter_mut()
                .zip(body_outputs.iter())
                .take(num_state_vars)
            {
                output.ty = body_output.ty.clone();
            }

            // Update scan output sequence types
            for i in num_state_vars..node.outputs.len() {
                let body_idx = i;
                if body_idx < body_outputs.len() {
                    match &body_outputs[body_idx].ty {
                        ArgType::Tensor(body_tensor) => {
                            node.outputs[i].ty = ArgType::Tensor(crate::ir::TensorType {
                                dtype: body_tensor.dtype,
                                rank: body_tensor.rank + 1,
                                static_shape: None,
                            });
                        }
                        _ => {
                            node.outputs[i].ty = body_outputs[body_idx].ty.clone();
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        // Extract body graph from attributes
        let body_attr = node
            .attrs
            .get("body")
            .ok_or_else(|| ProcessError::MissingAttribute("body".to_string()))?
            .clone();

        // Handle both Graph and GraphBuilder
        let body = match body_attr {
            crate::ir::AttributeValue::Graph(g) => g,
            crate::ir::AttributeValue::GraphBuilder(mut builder) => {
                // Convert NodeBuilders to Nodes
                let nodes = crate::ir::graph::finalize_graph_nodes(&mut builder.nodes, opset);
                crate::ir::OnnxGraph {
                    nodes,
                    inputs: std::mem::take(&mut builder.inputs),
                    outputs: std::mem::take(&mut builder.outputs),
                    _graph_data: builder._graph_data.clone(),
                }
            }
            _ => {
                return Err(ProcessError::Custom(
                    "Expected Graph or GraphBuilder for body".to_string(),
                ));
            }
        };

        // Extract num_scan_inputs (required)
        let num_scan_inputs = node
            .attrs
            .get("num_scan_inputs")
            .ok_or_else(|| ProcessError::MissingAttribute("num_scan_inputs".to_string()))?
            .clone()
            .into_i64();

        // Extract optional direction attributes
        let scan_input_directions = node
            .attrs
            .get("scan_input_directions")
            .map(|v| v.clone().into_i64s())
            .unwrap_or_default();

        let scan_output_directions = node
            .attrs
            .get("scan_output_directions")
            .map(|v| v.clone().into_i64s())
            .unwrap_or_default();

        let scan_input_axes = node
            .attrs
            .get("scan_input_axes")
            .map(|v| v.clone().into_i64s())
            .unwrap_or_default();

        let scan_output_axes = node
            .attrs
            .get("scan_output_axes")
            .map(|v| v.clone().into_i64s())
            .unwrap_or_default();

        Ok(ScanConfig {
            body,
            num_scan_inputs,
            scan_input_directions,
            scan_output_directions,
            scan_input_axes,
            scan_output_axes,
        })
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Scan {
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
    use crate::ir::{Argument, DType, NodeType, OnnxGraph, TensorType};
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_body(num_state_vars: usize, num_scan_inputs: usize) -> OnnxGraph {
        let mut body_inputs = Vec::new();
        let mut body_outputs = Vec::new();

        // State variable inputs and outputs
        for i in 0..num_state_vars {
            body_inputs.push(Argument {
                name: format!("state_{}", i),
                ty: ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 2,
                    static_shape: Some(vec![2, 3]),
                }),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            });

            body_outputs.push(Argument {
                name: format!("state_{}_out", i),
                ty: ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 2,
                    static_shape: Some(vec![2, 3]),
                }),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            });
        }

        // Scan input elements
        for i in 0..num_scan_inputs {
            body_inputs.push(Argument {
                name: format!("scan_in_{}", i),
                ty: ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 2,
                    static_shape: Some(vec![2, 3]),
                }),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            });
        }

        // Scan outputs
        for i in 0..num_scan_inputs {
            body_outputs.push(Argument {
                name: format!("scan_out_{}", i),
                ty: ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 2,
                    static_shape: Some(vec![2, 3]),
                }),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            });
        }

        OnnxGraph {
            nodes: vec![],
            inputs: body_inputs,
            outputs: body_outputs,
            _graph_data: None,
        }
    }

    #[test]
    fn test_scan_infer_types_basic() {
        let num_state_vars = 1;
        let num_scan_inputs = 1;
        let body = create_test_body(num_state_vars, num_scan_inputs);

        let mut node = TestNodeBuilder::new(NodeType::Scan, "test_scan")
            .input_tensor_f32("initial_state", 2, Some(vec![2, 3]))
            .input_tensor_f32("scan_input_seq", 3, Some(vec![4, 2, 3]))
            .build();

        node.attrs
            .insert("body".to_string(), AttributeValue::Graph(body));
        node.attrs
            .insert("num_scan_inputs".to_string(), AttributeValue::Int64(1));

        let processor = ScanProcessor;
        // Extract config first
        let _config = processor.extract_config(&node, 8).unwrap();

        let result = processor.infer_types(&mut node, 8, &OutputPreferences::default());

        assert!(result.is_ok());
        // Should have 2 outputs: final_state and scan_output_seq
        assert_eq!(node.outputs.len(), 2);
    }
}
