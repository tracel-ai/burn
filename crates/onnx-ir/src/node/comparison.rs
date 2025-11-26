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

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, DType, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Node representation for Equal operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct EqualNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node representation for Greater operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct GreaterNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node representation for GreaterOrEqual operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct GreaterOrEqualNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node representation for Less operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct LessNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node representation for LessOrEqual operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct LessOrEqualNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Update output type for comparison operations (e.g., Equal, Greater) to max input rank.
pub(crate) fn elementwise_comparison_outputs(node: &mut RawNode) {
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

pub(crate) struct ComparisonProcessor;

impl NodeProcessor for ComparisonProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 7,
            max_opset: None,
            inputs: InputSpec::Exact(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset based on operation type (individual validation for specific ops)
        let min_opset = match node.node_type {
            crate::ir::NodeType::Equal => 7,
            crate::ir::NodeType::Greater | crate::ir::NodeType::Less => 7,
            crate::ir::NodeType::GreaterOrEqual | crate::ir::NodeType::LessOrEqual => 12,
            _ => unreachable!(
                "ComparisonProcessor should only be called for comparison operations, got: {:?}",
                node.node_type
            ),
        };

        if opset < min_opset {
            return Err(ProcessError::UnsupportedOpset {
                required: min_opset,
                actual: opset,
            });
        }

        // TODO: Add validation for unexpected attributes - comparison ops should have no attributes
        // TODO: Add test for NaN comparison behavior - spec doesn't clearly specify Equal(NaN, NaN) result
        // TODO: Add test for inf comparisons - positive and negative infinity edge cases
        // TODO: Add test for bfloat16 support (opset 13+ for Equal, 16+ for GreaterOrEqual/LessOrEqual)
        // TODO: Add test for int4/uint4 types (opset 19+ for Equal) - mentioned in spec
        // TODO: Validate broadcasting rules are correctly applied for mismatched shapes

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

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        match builder.node_type {
            crate::ir::NodeType::Equal => Node::Equal(EqualNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            }),
            crate::ir::NodeType::Greater => Node::Greater(GreaterNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            }),
            crate::ir::NodeType::GreaterOrEqual => Node::GreaterOrEqual(GreaterOrEqualNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            }),
            crate::ir::NodeType::Less => Node::Less(LessNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            }),
            crate::ir::NodeType::LessOrEqual => Node::LessOrEqual(LessOrEqualNode {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            }),
            _ => panic!("ComparisonProcessor called with unsupported node type"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(input1_rank: usize, input2_rank: usize) -> RawNode {
        TestNodeBuilder::new(NodeType::Equal, "test_comparison")
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

    #[test]
    fn test_equal_opset_7() {
        let mut node = TestNodeBuilder::new(NodeType::Equal, "test_equal")
            .input_tensor_f32("A", 2, None)
            .input_tensor_f32("B", 2, None)
            .output_tensor_bool("result", 0, None)
            .build();

        let processor = ComparisonProcessor;
        let prefs = OutputPreferences::new();

        // Should work with opset 7
        assert!(processor.infer_types(&mut node, 7, &prefs).is_ok());

        // Should fail with opset 6
        let mut node = TestNodeBuilder::new(NodeType::Equal, "test_equal")
            .input_tensor_f32("A", 2, None)
            .input_tensor_f32("B", 2, None)
            .output_tensor_bool("result", 0, None)
            .build();
        assert!(processor.infer_types(&mut node, 6, &prefs).is_err());
    }

    #[test]
    fn test_greater_less_opset_7() {
        let processor = ComparisonProcessor;
        let prefs = OutputPreferences::new();

        // Test Greater with opset 7
        let mut node = TestNodeBuilder::new(NodeType::Greater, "test_greater")
            .input_tensor_f32("A", 2, None)
            .input_tensor_f32("B", 2, None)
            .output_tensor_bool("result", 0, None)
            .build();
        assert!(processor.infer_types(&mut node, 7, &prefs).is_ok());

        // Test Less with opset 7
        let mut node = TestNodeBuilder::new(NodeType::Less, "test_less")
            .input_tensor_f32("A", 2, None)
            .input_tensor_f32("B", 2, None)
            .output_tensor_bool("result", 0, None)
            .build();
        assert!(processor.infer_types(&mut node, 7, &prefs).is_ok());

        // Should fail with opset 6
        let mut node = TestNodeBuilder::new(NodeType::Greater, "test_greater")
            .input_tensor_f32("A", 2, None)
            .input_tensor_f32("B", 2, None)
            .output_tensor_bool("result", 0, None)
            .build();
        assert!(processor.infer_types(&mut node, 6, &prefs).is_err());
    }

    #[test]
    fn test_greater_or_equal_less_or_equal_opset_12() {
        let processor = ComparisonProcessor;
        let prefs = OutputPreferences::new();

        // Test GreaterOrEqual with opset 12
        let mut node = TestNodeBuilder::new(NodeType::GreaterOrEqual, "test_gte")
            .input_tensor_f32("A", 2, None)
            .input_tensor_f32("B", 2, None)
            .output_tensor_bool("result", 0, None)
            .build();
        assert!(processor.infer_types(&mut node, 12, &prefs).is_ok());

        // Test LessOrEqual with opset 12
        let mut node = TestNodeBuilder::new(NodeType::LessOrEqual, "test_lte")
            .input_tensor_f32("A", 2, None)
            .input_tensor_f32("B", 2, None)
            .output_tensor_bool("result", 0, None)
            .build();
        assert!(processor.infer_types(&mut node, 12, &prefs).is_ok());

        // Should fail with opset 11
        let mut node = TestNodeBuilder::new(NodeType::GreaterOrEqual, "test_gte")
            .input_tensor_f32("A", 2, None)
            .input_tensor_f32("B", 2, None)
            .output_tensor_bool("result", 0, None)
            .build();
        assert!(processor.infer_types(&mut node, 11, &prefs).is_err());
    }
}
