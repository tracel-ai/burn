//! # PRelu
//!
//! Applies the Parametric Rectified Linear Unit (PReLU) activation function element-wise.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__PRelu.html>
//!
//! ## Attributes
//! None (slope is provided as an input tensor)
//!
//! ## Inputs
//! - `X` (T): Input tensor of any shape
//! - `slope` (T): Slope tensor for negative values. Must be unidirectional broadcastable to X.
//!
//! ## Outputs
//! - `Y` (T): Output tensor with the same shape and type as X
//!
//! ## Type Constraints
//! - `T`: Constrained to numeric tensors (float16, float32, float64, bfloat16, uint8, uint16,
//!   uint32, uint64, int8, int16, int32, int64)
//!
//! ## Broadcasting
//! The slope tensor must be unidirectional broadcastable to the input tensor X.
//! Common patterns:
//! - Scalar slope (shape \[1\]): Same slope for all elements
//! - Per-channel slope (shape \[C, 1, 1\]): Different slope per channel
//!
//! ## Opset Versions
//! - **Opset 1-5**: Initial version
//! - **Opset 6**: Improved broadcasting semantics
//! - **Opset 7-8**: Updated shape inference
//! - **Opset 9-15**: Expanded type support
//! - **Opset 16+**: Added bfloat16 support
//!
//! ## Implementation Notes
//! - The slope input is lifted to static during constant lifting phase
//! - This allows the slope to be embedded in the generated code

use crate::ir::Node;
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

pub struct PReluProcessor;

impl NodeProcessor for PReluProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // Lift the slope input (input[1]) to static
        if node.inputs.len() > 1 {
            node.inputs[1].to_static()?;
        }
        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 6)?;
        crate::processor::validate_input_count(node, 2)?;
        crate::processor::validate_output_count(node, 1)?;

        // Validate no unexpected attributes
        if !node.attrs.is_empty() {
            let keys: Vec<String> = node.attrs.keys().cloned().collect();
            return Err(ProcessError::InvalidAttribute {
                name: keys[0].clone(),
                reason: format!("PRelu does not support any attributes, found: {:?}", keys),
            });
        }

        // Output type follows broadcasting rules (same as input X)
        crate::processor::same_as_input_broadcast(node);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, NodeType};
    use crate::node::test_utils::NodeBuilder;
    use burn_tensor::DType;

    fn create_test_node() -> Node {
        NodeBuilder::new(NodeType::PRelu, "test_prelu")
            .input_tensor_f32("X", 4, Some(vec![1, 3, 224, 224]))
            .input_tensor_f32("slope", 4, Some(vec![1, 3, 1, 1])) // Per-channel slope
            .output_tensor_f32("Y", 0, None) // Rank will be inferred
            .build()
    }

    #[test]
    fn test_prelu_type_inference() {
        let mut node = create_test_node();
        let processor = PReluProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Check output type matches input X
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 4);
                assert_eq!(tensor.static_shape, Some(vec![1, 3, 224, 224]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_prelu_no_attributes_allowed() {
        let mut node = create_test_node();
        node.attrs.insert(
            "invalid_attr".to_string(),
            crate::ir::AttributeValue::Float32(0.5),
        );
        let processor = PReluProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_prelu_scalar_slope() {
        let mut node = NodeBuilder::new(NodeType::PRelu, "test_prelu")
            .input_tensor_f32("X", 2, Some(vec![10, 20]))
            .input_tensor_f32("slope", 1, Some(vec![1])) // Scalar slope
            .output_tensor_f32("Y", 0, None)
            .build();

        let processor = PReluProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output shape should match input X
        if let ArgType::Tensor(tensor) = &node.outputs[0].ty {
            assert_eq!(tensor.static_shape, Some(vec![10, 20]));
        } else {
            panic!("Expected tensor output");
        }
    }

    #[test]
    fn test_prelu_requires_two_inputs() {
        let mut node = NodeBuilder::new(NodeType::PRelu, "test_prelu")
            .input_tensor_f32("X", 2, Some(vec![10, 20]))
            .output_tensor_f32("Y", 0, None)
            .build();

        let processor = PReluProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount { .. })
        ));
    }
}
