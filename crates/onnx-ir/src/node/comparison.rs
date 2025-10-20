//! # Comparison Operations (Equal, Greater, Less, GreaterOrEqual, LessOrEqual)
//!
//! Comparison operators perform element-wise comparisons between two input tensors and return
//! boolean tensors indicating the result of the comparison at each position. These operations
//! support broadcasting according to ONNX broadcasting rules.
//!
//! **ONNX Specs**:
//! - Equal: <https://onnx.ai/onnx/operators/onnx__Equal.html>
//! - Greater: <https://onnx.ai/onnx/operators/onnx__Greater.html>
//! - Less: <https://onnx.ai/onnx/operators/onnx__Less.html>
//! - GreaterOrEqual: <https://onnx.ai/onnx/operators/onnx__GreaterOrEqual.html>
//! - LessOrEqual: <https://onnx.ai/onnx/operators/onnx__LessOrEqual.html>
//!
//! ## Attributes
//!
//! None of these operations have attributes.
//!
//! ## Inputs
//!
//! - **A** (T): First operand tensor
//! - **B** (T): Second operand tensor
//!
//! ## Outputs
//!
//! - **C** (T1): Boolean tensor with the result of the element-wise comparison. The output shape
//!   follows broadcasting rules based on the input shapes.
//!
//! ## Opset Versions
//!
//! - **Equal**:
//!   - Opset 7-10: Initial version with basic type support
//!   - Opset 11-12: Extended type support for additional numeric types
//!   - Opset 13-18: Added bfloat16 support
//!   - Opset 19+: Added int4 and uint4 support
//!
//! - **Greater**:
//!   - Opset 7-8: Initial version with basic type support
//!   - Opset 9-12: Extended type support for additional numeric types
//!   - Opset 13+: Added bfloat16 support
//!
//! - **Less**:
//!   - Opset 7-8: Initial version with basic type support
//!   - Opset 9-12: Extended type support for additional numeric types
//!   - Opset 13+: Added bfloat16 support
//!
//! - **GreaterOrEqual**:
//!   - Opset 12-15: Initial version
//!   - Opset 16+: Added bfloat16 support
//!
//! - **LessOrEqual**:
//!   - Opset 12-15: Initial version
//!   - Opset 16+: Added bfloat16 support
//!
//! ## Implementation Notes
//!
//! - All comparison operations output boolean tensors (element type: bool)
//! - The output rank is determined by the maximum rank of the input tensors
//! - When both inputs are scalars, the output is a scalar boolean
//! - Special handling for Shape-to-Shape comparisons where the output is also a Shape type

use crate::ir::{ArgType, DType, Node, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

/// Update output type for comparison operations (e.g., Equal, Greater) to max input rank.
pub fn elementwise_comparison_outputs(node: &mut Node) {
    // Check if both inputs are Shape types
    let both_shapes = node.inputs.len() == 2
        && matches!(&node.inputs[0].ty, ArgType::Shape(_))
        && matches!(&node.inputs[1].ty, ArgType::Shape(_));

    if both_shapes {
        // For Shape-to-Shape comparison, output should be a Shape type
        // Get the dimension from the first Shape input
        if let ArgType::Shape(dim) = &node.inputs[0].ty {
            node.outputs[0].ty = ArgType::Shape(*dim);
            return;
        }
    }

    let max_rank = node.inputs.iter().fold(0, |acc, input| match &input.ty {
        ArgType::Tensor(tensor) => acc.max(tensor.rank),
        ArgType::Scalar(_) => acc,
        ArgType::Shape(_) => acc.max(1), // Shape types are always rank 1
    });

    if max_rank == 0 {
        node.outputs[0].ty = ArgType::Scalar(DType::Bool);
    } else {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            dtype: DType::Bool,
            rank: max_rank,
            static_shape: None,
        });
    }
}

pub struct ComparisonProcessor;

impl NodeProcessor for ComparisonProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset based on operation type
        let min_opset = match node.node_type {
            crate::ir::NodeType::Equal => 7,
            // FIXME: According to spec, Greater and Less should support opset 7+, but currently set to 9
            crate::ir::NodeType::Greater | crate::ir::NodeType::Less => 9,
            crate::ir::NodeType::GreaterOrEqual | crate::ir::NodeType::LessOrEqual => 12,
            // FIXME: Other comparison operations should not use default opset 1; need specific validation
            _ => 1,
        };

        crate::processor::validate_opset(opset, min_opset)?;
        crate::processor::validate_input_count(node, 2)?;
        crate::processor::validate_output_count(node, 1)?;

        // Check if both inputs are Shape types
        let both_shapes = node.inputs.len() == 2
            && matches!(&node.inputs[0].ty, ArgType::Shape(_))
            && matches!(&node.inputs[1].ty, ArgType::Shape(_));

        if both_shapes {
            // For Shape-to-Shape comparison, output should be a Shape type
            // Get the dimension from the first Shape input
            if let ArgType::Shape(dim) = &node.inputs[0].ty {
                node.outputs[0].ty = ArgType::Shape(*dim);
                return Ok(());
            }
        }

        let max_rank = node.inputs.iter().fold(0, |acc, input| match &input.ty {
            ArgType::Tensor(tensor) => acc.max(tensor.rank),
            ArgType::Scalar(_) => acc,
            ArgType::Shape(_) => acc.max(1), // Shape types are always rank 1
        });

        if max_rank == 0 {
            node.outputs[0].ty = ArgType::Scalar(DType::Bool);
        } else {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                dtype: DType::Bool,
                rank: max_rank,
                static_shape: None,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(input1_rank: usize, input2_rank: usize) -> Node {
        NodeBuilder::new(NodeType::Equal, "test_comparison")
            .input_tensor_f32("A", input1_rank, None)
            .input_tensor_f32("B", input2_rank, None)
            .output_tensor_bool("result", 0, None) // rank will be updated
            .build()
    }

    #[test]
    fn test_comparison_rank_broadcasting() {
        let mut node = create_test_node(2, 3);

        let processor = ComparisonProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::Bool);
                assert_eq!(tensor.rank, 3); // max(2, 3) = 3
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_comparison_scalar_result() {
        let mut node = create_test_node(0, 0);

        // Convert inputs to scalars
        node.inputs[0].ty = ArgType::Scalar(DType::F32);
        node.inputs[1].ty = ArgType::Scalar(DType::F32);

        let processor = ComparisonProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, DType::Bool);
            }
            _ => panic!("Expected scalar output"),
        }
    }

    #[test]
    fn test_comparison_with_shape_and_tensor() {
        let mut node = create_test_node(2, 2);
        node.inputs[0].ty = ArgType::Shape(3);
        // node.inputs[1] remains as Tensor with rank 2

        let processor = ComparisonProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::Bool);
                assert_eq!(tensor.rank, 2); // max(1, 2) = 2 (Shape is rank 1, Tensor is rank 2)
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_comparison_both_shape_inputs() {
        let mut node = create_test_node(0, 0);
        node.inputs[0].ty = ArgType::Shape(3);
        node.inputs[1].ty = ArgType::Shape(3);

        let processor = ComparisonProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Shape(dim) => {
                assert_eq!(*dim, 3); // Shape output with same dimension
            }
            _ => panic!("Expected shape output"),
        }
    }
}
