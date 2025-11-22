//! # LogSoftmax
//!
//! Computes log(softmax(x)) along specified axis.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__LogSoftmax.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with LogSoftmax operation.
//! - **Opset 11**: Changed default axis from 1 to -1 (last dimension); clarified axis behavior.
//! - **Opset 13**: Added bfloat16 type support; no functional changes to operation semantics.
//!
//! **Implementation Note**: This implementation validates opset 13+.
//!
//! ## Missing Test Coverage
//! - TODO: No test for axis=0 or positive axis values - Only axis=-1 tested
//! - TODO: No test for negative axis normalization edge cases - e.g., axis=-rank should map to 0
//! - TODO: No test for 1D tensors - Simplest case not tested
//! - TODO: No test for higher-rank tensors (4D, 5D) - Only 2D tested
//! - TODO: No test validating numerical stability with extreme values (very large/small inputs)
//! - TODO: No test for all-zero or constant inputs - Edge cases for softmax normalization
//! - TODO: No test validating that input must be floating-point type - Integer inputs should be rejected

use crate::ir::{ArgType, Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use derive_new::new;
use onnx_ir_derive::NodeBuilderDerive;

/// Configuration for LogSoftmax operations
#[derive(Debug, Clone, new)]
pub struct LogSoftmaxConfig {
    /// Axis along which to apply log softmax
    pub axis: usize,
}

/// Node representation for LogSoftmax operation
#[derive(Debug, Clone, NodeBuilderDerive)]
pub struct LogSoftmaxNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: LogSoftmaxConfig,
}

pub(crate) struct LogSoftmaxProcessor;

impl NodeProcessor for LogSoftmaxProcessor {
    type Config = LogSoftmaxConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 13,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Validate input tensor dtype is floating-point type - Type constraint not enforced - burn/crates/onnx-ir/src/node/log_softmax.rs:54

        // TODO: Validate unexpected attributes before config extraction
        // The spec only supports "axis" attribute
        for (key, _value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for LogSoftmax: {}", key),
                    });
                }
            }
        }

        // Infer output type
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        // Extract the shape of the input tensor
        let tensor = match &node.inputs.first().unwrap().ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs.first().unwrap().ty),
                });
            }
        };

        // Extract the axis attribute (default: -1 per ONNX spec)
        let mut axis: i64 = -1;

        for (key, value) in node.attrs.iter() {
            if key.as_str() == "axis" {
                axis = value.clone().into_i64()
            }
        }

        // if axis is negative, it is counted from the end
        if axis < 0 {
            axis += tensor.rank as i64;
        }

        // TODO: Validate converted axis is within bounds [0, rank) - Out of bounds axis should be rejected - burn/crates/onnx-ir/src/node/log_softmax.rs:103

        let config = LogSoftmaxConfig {
            axis: axis as usize,
        };
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::LogSoftmax(LogSoftmaxNode {
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
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(axis: i64, input_rank: usize) -> NodeBuilder {
        TestNodeBuilder::new(NodeType::LogSoftmax, "test_log_softmax")
            .input_tensor_f32("data", input_rank, None)
            .output_tensor_f32("output", input_rank, None)
            .attr_int("axis", axis)
            .build()
    }

    #[test]
    fn test_log_softmax_config_basic() {
        let node = create_test_node(-1, 3);
        let mut node = node;
        let processor = LogSoftmaxProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.axis, 2); // -1 + 3 = 2 (last dimension)
    }

    #[test]
    fn test_log_softmax_config_explicit_axis() {
        let node = create_test_node(1, 3);
        let mut node = node;
        let processor = LogSoftmaxProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_log_softmax_config_multiple_inputs() {
        let mut node = create_test_node(1, 3);
        // Add an extra input
        let extra_input = TestNodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_f32("extra", 1, None)
            .build()
            .inputs
            .pop()
            .unwrap();
        node.inputs.push(extra_input);
        let processor = LogSoftmaxProcessor;
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 1,
                actual: 2
            })
        ));
    }
}
