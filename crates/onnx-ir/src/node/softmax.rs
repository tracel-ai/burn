//! # Softmax
//!
//! Applies the Softmax activation function along a specified axis.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Softmax.html>
//!
//! ## Attributes
//! - `axis` (int, default=-1): The dimension along which Softmax will be performed.
//!   Negative values mean counting dimensions from the back. Accepted range is [-r, r-1]
//!   where r = rank(input).
//!
//! ## Inputs
//! - `input` (T): Input tensor of rank >= axis
//!
//! ## Outputs
//! - `output` (T): Output tensor with the same shape as the input tensor
//!
//! ## Type Constraints
//! - T: tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with axis=1 default, operates on 2D tensors.
//! - **Opset 11**: Changed default axis to -1 (last dimension). Maintains backward compatibility with 2D coercion behavior.
//! - **Opset 13**: Removed 2D coercion behavior. Softmax now operates along specified axis directly without reshaping. This is the current behavior.
//!
//! **Implementation Note**: This implementation requires opset 13+ and uses the modern behavior (no 2D coercion). The axis attribute defaults to -1 as per opset 11+ specification.

use crate::ir::{ArgType, Node, NodeConfig};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use std::any::Any;

/// Configuration for Softmax operations
#[derive(Debug, Clone)]
pub struct SoftmaxConfig {
    /// Axis along which to apply softmax
    pub axis: usize,
}

impl NodeConfig for SoftmaxConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct SoftmaxProcessor;

impl NodeProcessor for SoftmaxProcessor {
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
        node: &mut Node,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // FIXME: The spec requires the input rank to be >= 1 for the axis attribute to be valid.
        // The implementation should validate that the tensor rank is at least 1.
        // Edge case: what happens with a scalar (rank-0) input? Should be rejected.

        // TODO: Missing validation that axis is in valid range [-rank, rank-1].
        // Out-of-bounds axis values (after negative index handling) should be rejected.

        // Infer output type
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
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

        let config = SoftmaxConfig {
            axis: axis as usize,
        };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axis: i64, input_rank: usize) -> Node {
        NodeBuilder::new(NodeType::Softmax, "test_softmax")
            .input_tensor_f32("data", input_rank, None)
            .output_tensor_f32("output", input_rank, None)
            .attr_int("axis", axis)
            .build()
    }

    #[test]
    fn test_softmax_config_basic() {
        let node = create_test_node(-1, 3);
        let mut node = node;
        let processor = SoftmaxProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<SoftmaxConfig>();
        assert_eq!(config.axis, 2); // -1 + 3 = 2 (last dimension)
    }

    #[test]
    fn test_softmax_config_explicit_axis() {
        let node = create_test_node(1, 3);
        let mut node = node;
        let processor = SoftmaxProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<SoftmaxConfig>();
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_softmax_config_multiple_inputs() {
        let mut node = create_test_node(1, 3);
        // Add an extra input
        let extra_input = NodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_f32("extra", 1, None)
            .build()
            .inputs
            .pop()
            .unwrap();
        node.inputs.push(extra_input);
        let processor = SoftmaxProcessor;
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

    // TODO: Missing test for scalar (rank-0) input - should be rejected as rank must be >= 1.

    // TODO: Missing test for axis out of range - e.g., axis=5 for rank-3 tensor.

    // TODO: Missing test for opset 13 behavior change - spec changed from 2D coercion to direct axis operation.
    // Need test to verify opset < 13 is rejected and opset 13+ works correctly.

    // TODO: Missing test for type constraints - Softmax only supports float types.
    // Need test to verify integer input is rejected (or properly handled).

    // TODO: Missing test for 1D tensor with axis=0 - simplest valid case not tested.

    // TODO: Missing test for negative axis normalization - axis=-1 should work correctly.
    // Current test has this but doesn't verify the actual behavior, only config extraction.
}
