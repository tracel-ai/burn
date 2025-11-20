//! # Relu
//!
//! Applies the Rectified Linear Unit (ReLU) activation function element-wise.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Relu.html>
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

use crate::ir::{Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Node representation for Relu operation
#[derive(Debug, Clone)]
pub struct ReluNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

pub(crate) struct ReluProcessor;

impl NodeProcessor for ReluProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 6,
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
        // TODO: Missing test coverage for different data types
        // Tests only use f32. Spec supports float16, float64, bfloat16, and integer types.
        // Add tests: relu_float64, relu_int32, relu_int64
        // Note: Integer types may require checking if Burn backend supports ReLU on integers.

        // TODO: Missing test coverage for edge cases
        // No tests for:
        // - All negative inputs (output should be all zeros)
        // - All positive inputs (output should equal input)
        // - Mixed with exact zeros
        // - Very large/small values (numerical stability)
        // Add tests: relu_all_negative, relu_all_positive, relu_with_zeros

        // TODO: Missing test coverage for different tensor ranks
        // Tests cover 2D and 4D. Add coverage for 1D, 3D, 5D tensors.
        // Add tests: relu_1d, relu_3d, relu_5d

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

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        Node::Relu(ReluNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;
    use burn_tensor::DType;

    fn create_test_node() -> NodeBuilder {
        TestNodeBuilder::new(NodeType::Relu, "test_relu")
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
        let mut node = TestNodeBuilder::new(NodeType::Relu, "test_relu")
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
