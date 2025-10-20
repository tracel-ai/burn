//! # Relu
//!
//! Applies the Rectified Linear Unit (ReLU) activation function element-wise.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Relu.html>
//!
//! ## Attributes
//! None
//!
//! ## Inputs
//! - `X` (T): Input tensor of any shape
//!
//! ## Outputs
//! - `Y` (T): Output tensor with the same shape and type as input
//!
//! ## Type Constraints
//! - `T`: Constrained to numeric tensors (float16, float32, float64, bfloat16, int8,
//!   int16, int32, int64, uint8, uint16, uint32, uint64)
//!
//! ## Opset Versions
//! - **Opset 1-5**: Initial version
//! - **Opset 6-12**: Improved shape inference
//! - **Opset 13**: Expanded type support
//! - **Opset 14+**: Added bfloat16 support

use crate::ir::Node;
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

pub struct ReluProcessor;

impl NodeProcessor for ReluProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 6)?;
        crate::processor::validate_input_count(node, 1)?;
        crate::processor::validate_output_count(node, 1)?;

        // Validate no unexpected attributes
        if !node.attrs.is_empty() {
            let keys: Vec<String> = node.attrs.keys().cloned().collect();
            return Err(ProcessError::InvalidAttribute {
                name: keys[0].clone(),
                reason: format!("Relu does not support any attributes, found: {:?}", keys),
            });
        }

        // Output type is same as input
        crate::processor::same_as_input(node);

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
        NodeBuilder::new(NodeType::Relu, "test_relu")
            .input_tensor_f32("X", 4, Some(vec![1, 3, 224, 224]))
            .output_tensor_f32("Y", 0, None) // Rank will be inferred
            .build()
    }

    #[test]
    fn test_relu_type_inference() {
        let mut node = create_test_node();
        let processor = ReluProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Check output type matches input
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
    fn test_relu_no_attributes_allowed() {
        let mut node = create_test_node();
        node.attrs.insert(
            "invalid_attr".to_string(),
            crate::ir::AttributeValue::Float32(0.5),
        );
        let processor = ReluProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::InvalidAttribute { .. })));
    }

    #[test]
    fn test_relu_shape_preservation() {
        let mut node = NodeBuilder::new(NodeType::Relu, "test_relu")
            .input_tensor_f32("X", 2, Some(vec![10, 20]))
            .output_tensor_f32("Y", 0, None)
            .build();

        let processor = ReluProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        if let ArgType::Tensor(tensor) = &node.outputs[0].ty {
            assert_eq!(tensor.static_shape, Some(vec![10, 20]));
        } else {
            panic!("Expected tensor output");
        }
    }
}
