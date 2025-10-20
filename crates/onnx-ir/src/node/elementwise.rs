//! # Element-wise Operations
//!
//! Processors for element-wise unary and binary operations that operate on tensors element-by-element.
//!
//! **ONNX Specs**: Multiple operations with varying opset requirements
//!
//! ## Unary Operations
//!
//! **Common Pattern:**
//! - Single input tensor
//! - Single output tensor with same shape and type
//! - No attributes (except Round which has an optional mode attribute)
//! - Output shape identical to input shape
//!
//! **Opset 6+ Operations:**
//! - **Abs**: Absolute value |x|
//! - **Ceil**: Round up to nearest integer
//! - **Floor**: Round down to nearest integer
//! - **Exp**: Exponential e^x
//! - **Log**: Natural logarithm ln(x)
//! - **Neg**: Negation -x
//! - **Reciprocal**: Reciprocal 1/x
//! - **Sqrt**: Square root √x
//!
//! **Opset 7+ Operations (Trigonometric):**
//! - **Acos**: Arc cosine
//! - **Asin**: Arc sine
//! - **Atan**: Arc tangent
//! - **Cos**: Cosine
//! - **Sin**: Sine
//! - **Tan**: Tangent
//!
//! **Opset 9+ Operations:**
//! - **Erf**: Error function
//! - **Sign**: Sign function (-1, 0, or 1)
//!
//! **Opset 11+ Operations:**
//! - **Round**: Round to nearest integer
//!
//! **Opset 1+ Operations:**
//! - **Not**: Logical NOT
//!
//! ## Binary Operations
//!
//! **Common Pattern:**
//! - Two input tensors (supports broadcasting)
//! - Single output tensor
//! - No attributes (except PRelu which has a slope parameter)
//! - Output shape follows standard ONNX broadcasting semantics
//!
//! **Supported Operations:**
//! - **Pow**: Element-wise power a^b
//! - **Max**: Element-wise maximum
//! - **Min**: Element-wise minimum
//! - **And**: Logical AND
//! - **Or**: Logical OR
//! - **Xor**: Logical XOR
//! - **BitwiseAnd**: Bitwise AND
//! - **BitwiseOr**: Bitwise OR
//! - **BitwiseXor**: Bitwise XOR
//! - **PRelu**: Parametric ReLU (slope input is lifted to static)
//!
//! ## Type Inference
//!
//! - **Unary**: Output type and shape identical to input
//! - **Binary**: Output type matches inputs, shape follows broadcasting rules
//! - **PRelu**: Slope parameter (second input) is lifted to static constant

use crate::ir::Node;
use crate::processor::{
    NodeProcessor, OutputPreferences, ProcessError, same_as_input, same_as_input_broadcast,
};

/// Node processor for element-wise binary operations with broadcasting
///
/// Used for simple binary operations that don't require special type propagation:
/// - **Pow**: Element-wise power (a^b)
/// - **Max**: Element-wise maximum
/// - **Min**: Element-wise minimum
/// - **And**: Logical AND
/// - **Or**: Logical OR
/// - **Xor**: Logical XOR
/// - **BitwiseAnd**: Bitwise AND
/// - **BitwiseOr**: Bitwise OR
/// - **BitwiseXor**: Bitwise XOR
/// - **PRelu**: Parametric ReLU
///
/// These operations support standard ONNX broadcasting semantics without
/// needing Shape or Scalar type propagation (unlike arithmetic operations).
pub struct ElementwiseBinaryProcessor;

impl NodeProcessor for ElementwiseBinaryProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        // For PRelu, lift the slope input (input[1])
        if node.node_type == crate::ir::NodeType::PRelu && node.inputs.len() > 1 {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut Node,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: No opset validation - different binary ops have different minimum opsets
        // (Pow, Max, Min, And, Or, Xor, BitwiseAnd, BitwiseOr, BitwiseXor, PRelu)
        crate::processor::validate_input_count(node, 2)?;
        crate::processor::validate_output_count(node, 1)?;

        same_as_input_broadcast(node);
        Ok(())
    }
}

/// Node processor for element-wise unary operations
/// Used for: Neg, Abs, Ceil, Floor, Sqrt, Exp, Log, Sin, Cos, etc.
pub struct ElementwiseUnaryProcessor;

impl NodeProcessor for ElementwiseUnaryProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset based on operation type
        let min_opset = match node.node_type {
            // Opset 6 operations (shape inference improvements)
            crate::ir::NodeType::Abs
            | crate::ir::NodeType::Ceil
            | crate::ir::NodeType::Floor
            | crate::ir::NodeType::Exp
            | crate::ir::NodeType::Log
            | crate::ir::NodeType::Neg
            | crate::ir::NodeType::Reciprocal
            | crate::ir::NodeType::Sqrt => 6,
            // Opset 7 operations (trigonometric functions)
            crate::ir::NodeType::Acos
            | crate::ir::NodeType::Asin
            | crate::ir::NodeType::Atan
            | crate::ir::NodeType::Cos
            | crate::ir::NodeType::Sin
            | crate::ir::NodeType::Tan => 7,
            // Opset 9 operations
            crate::ir::NodeType::Erf | crate::ir::NodeType::Sign => 9,
            // Opset 11 operations
            crate::ir::NodeType::Round => 11,
            // Opset 1 operations
            crate::ir::NodeType::Not => 1,
            // Other unary operations
            _ => 1, // FIXME: Default case should not be needed - all unary ops should be explicitly listed
        };

        crate::processor::validate_opset(opset, min_opset)?;
        crate::processor::validate_input_count(node, 1)?;
        crate::processor::validate_output_count(node, 1)?;

        // TODO: Spec mentions Round has optional 'mode' attribute - not validated or extracted
        same_as_input(node);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, NodeType, TensorType};

    #[test]
    fn test_elementwise_binary_processor() {
        let processor = ElementwiseBinaryProcessor;
        let prefs = OutputPreferences::new();

        let mut node = crate::ir::Node {
            node_type: NodeType::Max,
            name: "test_max".to_string(),
            inputs: vec![
                Argument {
                    name: "a".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 2,
                        static_shape: None,
                    }),
                    data_id: None,
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 2,
                        static_shape: None,
                    }),
                    data_id: None,
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
            ],
            outputs: vec![Argument {
                name: "c".to_string(),
                ty: ArgType::default(),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 2),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_elementwise_unary_processor() {
        let processor = ElementwiseUnaryProcessor;
        let prefs = OutputPreferences::new();

        let mut node = crate::ir::Node {
            node_type: NodeType::Neg,
            name: "test_neg".to_string(),
            inputs: vec![Argument {
                name: "a".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 3,
                    static_shape: None,
                }),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            outputs: vec![Argument {
                name: "b".to_string(),
                ty: ArgType::default(),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should match input
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 3);
                assert_eq!(t.elem_type, ElementType::Float32);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_unary_unsupported_opset() {
        let processor = ElementwiseUnaryProcessor;
        let prefs = OutputPreferences::new();

        let mut node = crate::ir::Node {
            node_type: NodeType::Round,
            name: "test_round".to_string(),
            inputs: vec![Argument {
                name: "a".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 2,
                    static_shape: None,
                }),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            outputs: vec![Argument {
                name: "b".to_string(),
                ty: ArgType::default(),
                data_id: None,
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        let result = processor.infer_types(&mut node, 10, &prefs);
        assert!(matches!(
            result,
            Err(ProcessError::UnsupportedOpset {
                required: 11,
                actual: 10
            })
        ));
    }
}
