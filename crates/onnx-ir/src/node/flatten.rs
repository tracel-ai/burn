//! # Flatten
//!
//! Flattens input tensor into a 2D matrix by splitting at a specified axis.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Flatten.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version with basic flatten operation.
//! - **Opset 9**: No functional changes (extended type support).
//! - **Opset 11**: Added support for negative axis values.
//! - **Opset 13**: Extended type constraints (added bfloat16 support).
//!
//! **Implementation Note**: This implementation validates opset 9+ (see FIXME at line 49).

use crate::ir::{ArgType, NodeBuilder, NodeConfig, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use std::any::Any;

/// Configuration for Flatten operations
#[derive(Debug, Clone)]
pub struct FlattenConfig {
    /// Axis along which to flatten
    pub axis: usize,
}

impl NodeConfig for FlattenConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct FlattenProcessor;

impl NodeProcessor for FlattenProcessor {
    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
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

        // check if the input tensor has at least 2 dimensions
        if tensor.rank < 2 {
            return Err(ProcessError::Custom(format!(
                "Flatten: input tensor must have at least 2 dimensions (got {})",
                tensor.rank
            )));
        }

        // Get reference to config for type inference
        let _config = node.config::<FlattenConfig>();

        // Infer output type - Flatten to a 2D tensor
        node.outputs[0].ty = ArgType::Tensor(TensorType { rank: 2, ..tensor });

        Ok(())
    }

    fn extract_config(
        &self,
        node: &NodeBuilder,
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

        // Extract the axis attribute (default: 1 per ONNX spec)
        let mut axis: i64 = 1;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => axis = value.clone().into_i64(),
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for Flatten: {}", key),
                    });
                }
            }
        }

        // if axis is negative, it is counted from the end
        if axis < 0 {
            axis += tensor.rank as i64;
        }

        // TODO: Validate axis is within valid range [0, rank) after normalization - Invalid axis values should return error - Missing range validation
        // TODO: Validate negative axis support for opset < 11 - Negative axis added in opset 11, should error for earlier opsets - Missing opset-specific validation

        let config = FlattenConfig {
            axis: axis as usize,
        };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(axis: i64) -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Flatten, "test_flatten")
            .input_tensor_f32("data", 4, None)
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", axis)
    }

    #[test]
    fn test_flatten_config_basic() {
        let node = create_test_node(1).process(FlattenProcessor, 16);
        let config = node.config::<FlattenConfig>();
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_flatten_config_with_negative_axis() {
        let node = create_test_node(-2).process(FlattenProcessor, 16);
        let config = node.config::<FlattenConfig>();
        assert_eq!(config.axis, 2); // -2 + 4 = 2
    }

    #[test]
    fn test_flatten_config_with_low_rank() {
        let mut node = create_test_node(1).build();
        // Replace the input with one that has lower rank
        let input = TestNodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_f32("x", 1, None)
            .build()
            .inputs
            .pop()
            .unwrap();
        node.inputs[0] = input;

        let processor = FlattenProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::Custom(_))));
    }

    #[test]
    fn test_flatten_config_with_multiple_inputs() {
        let mut node = create_test_node(1).build();
        // Add an extra input
        let extra_input = TestNodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_f32("extra", 1, None)
            .build()
            .inputs
            .pop()
            .unwrap();
        node.inputs.push(extra_input);

        let processor = FlattenProcessor;
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

    // TODO: Add test for axis out of range - Test axis >= rank should return error - Missing constraint validation test
    // TODO: Add test for negative axis with opset < 11 - Should fail per spec, negative axis added in opset 11 - Missing opset validation test
    // TODO: Add test for axis=0 edge case - Flattens entire tensor to 1D then reshapes to (1, N) - Missing edge case test
    // TODO: Add test for axis=rank edge case - Should produce (N, 1) output - Missing edge case test
    // TODO: Add test for static shape preservation - Should compute output static shape when input has static shape - Missing shape inference test
    // TODO: Add test for different data types - Spec supports all data types, not just f32 - Missing type coverage
    // TODO: Add test for unexpected attributes - Should reject unknown attributes per implementation - Missing attribute validation test
}
