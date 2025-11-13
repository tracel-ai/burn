//! # Element-wise Operations
//!
//! Processors for element-wise unary and binary operations that operate on tensors element-by-element.
//!
//! **ONNX Specs**: Multiple operations with varying opset requirements
//!
//! ## Opset Versions
//!
//! **Opset 1 Operations:**
//! - **Not**: Logical NOT
//!
//! **Opset 6 Operations:**
//! - **Abs**: Absolute value |x| (improved shape inference)
//! - **Ceil**: Round up to nearest integer (improved shape inference)
//! - **Floor**: Round down to nearest integer (improved shape inference)
//! - **Exp**: Exponential e^x (improved shape inference)
//! - **Log**: Natural logarithm ln(x) (improved shape inference)
//! - **Neg**: Negation -x (improved shape inference)
//! - **Reciprocal**: Reciprocal 1/x (improved shape inference)
//! - **Sqrt**: Square root √x (improved shape inference)
//!
//! **Opset 7 Operations (Trigonometric):**
//! - **Acos**: Arc cosine (domain [-1, 1])
//! - **Asin**: Arc sine (domain [-1, 1])
//! - **Atan**: Arc tangent
//! - **Cos**: Cosine
//! - **Sin**: Sine
//! - **Tan**: Tangent
// TODO: Missing test coverage for domain errors - Acos/Asin require input in [-1,1], Atanh in (-1,1), Acosh >= 1 - No tests validate behavior outside valid domain - Need edge case tests
//!
//! **Opset 9 Operations:**
//! - **Erf**: Error function
//! - **Sign**: Sign function (-1, 0, or 1)
//! - **Sinh**: Hyperbolic sine
//! - **Cosh**: Hyperbolic cosine
//! - **Tanh**: Hyperbolic tangent
//! - **Asinh**: Inverse hyperbolic sine
//! - **Acosh**: Inverse hyperbolic cosine (domain [1, ∞))
//! - **Atanh**: Inverse hyperbolic tangent (domain (-1, 1))
//!
//! **Opset 11 Operations:**
//! - **Round**: Round to nearest integer (supports optional mode attribute)
//!
//! **Other Operations:**
//! - **Sigmoid**: Sigmoid function 1/(1+e^-x) (Opset 6+)
//! - **Gelu**: Gaussian Error Linear Unit (Opset 20+)
// TODO: Gelu supports 'approximate' attribute (default="none", also "tanh") per ONNX spec - Not extracted or validated - Should add config extraction
//! - **Pow**: Element-wise power a^b (Opset 1+)
//! - **Max**: Element-wise maximum (Opset 1+)
//! - **Min**: Element-wise minimum (Opset 1+)
//! - **And**: Logical AND (Opset 1+)
//! - **Or**: Logical OR (Opset 1+)
//! - **Xor**: Logical XOR (Opset 1+)
//! - **BitwiseAnd**: Bitwise AND (Opset 18+)
//! - **BitwiseOr**: Bitwise OR (Opset 18+)
//! - **BitwiseXor**: Bitwise XOR (Opset 18+)
//! - **BitwiseNot**: Bitwise NOT (Opset 18+)
//!
//! ## Implementation Notes
//! - No opset validation currently performed for binary operations (see TODO at line 108)

use crate::ir::{Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
    same_as_input_broadcast,
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
///
/// These operations support standard ONNX broadcasting semantics without
/// needing Shape or Scalar type propagation (unlike arithmetic operations).
pub struct ElementwiseBinaryProcessor;

impl NodeProcessor for ElementwiseBinaryProcessor {
    type Config = ();

    fn extract_config(
        &self,
        _node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        Ok(())
    }

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: No opset validation - different binary ops have different minimum opsets
        // (Pow, Max, Min, And, Or, Xor, BitwiseAnd, BitwiseOr, BitwiseXor)
        // TODO: BitwiseAnd/BitwiseOr/BitwiseXor require opset 18+ per ONNX spec - Missing opset validation - Should validate minimum opset version
        // TODO: Validate no unexpected attributes for binary operations - Per spec, these ops have no attributes - Missing attribute validation
        // TODO: Max and Min support variadic inputs (2+ inputs) per ONNX spec, not just 2 - Missing support for multiple inputs - Should use validate_min_inputs instead

        same_as_input_broadcast(node);
        Ok(())
    }

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        use crate::ir::NodeType;

        match builder.node_type {
            NodeType::Pow => Node::Pow {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Max => Node::Max {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Min => Node::Min {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::And => Node::And {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Or => Node::Or {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Xor => Node::Xor {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::BitwiseAnd => Node::BitwiseAnd {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::BitwiseOr => Node::BitwiseOr {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::BitwiseXor => Node::BitwiseXor {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            _ => panic!(
                "Unsupported node type for ElementwiseBinaryProcessor: {:?}",
                builder.node_type
            ),
        }
    }
}

/// Node processor for element-wise unary operations
/// Used for: Neg, Abs, Ceil, Floor, Sqrt, Exp, Log, Sin, Cos, etc.
pub struct ElementwiseUnaryProcessor;

impl NodeProcessor for ElementwiseUnaryProcessor {
    type Config = ();

    fn extract_config(
        &self,
        _node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        Ok(())
    }

    fn spec(&self) -> NodeSpec {
        // Determine opset based on operation type
        let min_opset = 1;

        NodeSpec {
            min_opset,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
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
            // TODO: Hyperbolic trig functions (Sinh, Cosh, Tanh) introduced in opset 9 per ONNX spec - Missing from opset validation - Should add to opset 9 operations
            // TODO: Inverse hyperbolic functions (Asinh, Acosh, Atanh) introduced in opset 9 per ONNX spec - Missing from opset validation - Should add to opset 9 operations
            // TODO: Sigmoid introduced in opset 6, Gelu in opset 20 - Missing from opset validation - Should add proper opset checks
            // Opset 9 operations
            crate::ir::NodeType::Erf
            | crate::ir::NodeType::Sign
            | crate::ir::NodeType::Sinh
            | crate::ir::NodeType::Cosh
            | crate::ir::NodeType::Tanh
            | crate::ir::NodeType::Asinh
            | crate::ir::NodeType::Acosh
            | crate::ir::NodeType::Atanh => 9,
            // Opset 11 operations
            crate::ir::NodeType::Round => 11,
            // Opset 1 operations
            crate::ir::NodeType::Not => 1,
            // Opset 18 operations (bitwise)
            crate::ir::NodeType::BitwiseNot => 18,
            // Other unary operations (need proper opset validation)
            crate::ir::NodeType::Sigmoid => 6,
            crate::ir::NodeType::Gelu => 20, // TODO: Verify Gelu opset requirement - may need custom processor for 'approximate' attribute
            crate::ir::NodeType::GlobalAveragePool => 1, // TODO: GlobalAveragePool should not be in elementwise processor - has specific pooling semantics - Should have dedicated processor
            // Other unary operations
            _ => 1, // FIXME: Default case should not be needed - all unary ops should be explicitly listed
        };

        crate::processor::validate_opset(opset, min_opset)?;

        // TODO: Spec mentions Round has optional 'mode' attribute - not validated or extracted
        // TODO: Validate no unexpected attributes for unary operations except Round - Most unary ops have no attributes - Missing attribute validation
        // TODO: Hyperbolic functions (Sinh, Cosh, Tanh, Asinh, Acosh, Atanh) not listed in opset validation - May be missing from elementwise processor - Check if handled elsewhere
        same_as_input(node);
        Ok(())
    }

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        use crate::ir::NodeType;

        match builder.node_type {
            NodeType::Abs => Node::Abs {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Ceil => Node::Ceil {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Floor => Node::Floor {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Exp => Node::Exp {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Log => Node::Log {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Neg => Node::Neg {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Reciprocal => Node::Reciprocal {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Sqrt => Node::Sqrt {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Acos => Node::Acos {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Asin => Node::Asin {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Atan => Node::Atan {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Cos => Node::Cos {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Sin => Node::Sin {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Tan => Node::Tan {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Erf => Node::Erf {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Sign => Node::Sign {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Sinh => Node::Sinh {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Cosh => Node::Cosh {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Tanh => Node::Tanh {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Asinh => Node::Asinh {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Acosh => Node::Acosh {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Atanh => Node::Atanh {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Round => Node::Round {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Not => Node::Not {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::BitwiseNot => Node::BitwiseNot {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Sigmoid => Node::Sigmoid {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::Gelu => Node::Gelu {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            NodeType::GlobalAveragePool => Node::GlobalAveragePool {
                name: builder.name,
                inputs: builder.inputs,
                outputs: builder.outputs,
            },
            _ => panic!(
                "Unsupported node type for ElementwiseUnaryProcessor: {:?}",
                builder.node_type
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, DType, NodeType, TensorType};

    #[test]
    fn test_elementwise_binary_processor() {
        let processor = ElementwiseBinaryProcessor;
        let prefs = OutputPreferences::new();

        let mut node = crate::ir::NodeBuilder {
            node_type: NodeType::Max,
            name: "test_max".to_string(),
            inputs: vec![
                Argument {
                    name: "a".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        dtype: DType::F32,
                        rank: 2,
                        static_shape: None,
                    }),
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        dtype: DType::F32,
                        rank: 2,
                        static_shape: None,
                    }),
                    value_source: crate::ir::ValueSource::Dynamic,
                    value_store: None,
                },
            ],
            outputs: vec![Argument {
                name: "c".to_string(),
                ty: ArgType::default(),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
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

        let mut node = crate::ir::NodeBuilder {
            node_type: NodeType::Neg,
            name: "test_neg".to_string(),
            inputs: vec![Argument {
                name: "a".to_string(),
                ty: ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 3,
                    static_shape: None,
                }),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            outputs: vec![Argument {
                name: "b".to_string(),
                ty: ArgType::default(),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
        };

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should match input
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 3);
                assert_eq!(t.dtype, DType::F32);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_unary_unsupported_opset() {
        let processor = ElementwiseUnaryProcessor;
        let prefs = OutputPreferences::new();

        let mut node = crate::ir::NodeBuilder {
            node_type: NodeType::Round,
            name: "test_round".to_string(),
            inputs: vec![Argument {
                name: "a".to_string(),
                ty: ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 2,
                    static_shape: None,
                }),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            outputs: vec![Argument {
                name: "b".to_string(),
                ty: ArgType::default(),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
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

    // TODO: Add test for Max/Min with more than 2 inputs - Spec supports variadic inputs - Missing test for variadic input support
    // TODO: Add test for BitwiseAnd/BitwiseOr/BitwiseXor with opset < 18 - Should fail per spec - Missing opset validation test
    // TODO: Add test for Round with mode attribute - Round supports 'mode' attribute (default="nearest_round_half_to_even") - Missing attribute test
    // TODO: Add test for broadcasting behavior - Binary ops should support standard ONNX broadcasting - Missing broadcast test
    // TODO: Add test for type constraints - Different ops have different allowed types (e.g., Pow supports numeric types, And/Or/Xor support bool only) - Missing type validation test
    // TODO: Add test for zero-size tensors - Edge case where tensor has 0 elements - Missing edge case test
    // TODO: Add test for all unary operations listed - Only testing Neg and Round, missing Abs, Ceil, Floor, Sqrt, Exp, Log, trig functions, etc. - Incomplete test coverage
    // TODO: Add test for unexpected attributes on ops without attributes - Should reject unknown attributes per spec - Missing attribute validation test
    // TODO: Missing test coverage for inverse trig functions - No tests for Asin, Acos, Atan in onnx-tests - Need test cases
    // TODO: Missing test coverage for inverse hyperbolic functions - No tests for Asinh, Acosh, Atanh in onnx-tests - Need test cases
    // TODO: Missing test coverage for Reciprocal edge cases - No tests for division by zero behavior - Need edge case tests
    // TODO: Missing test for Reciprocal with different dtypes - Test only covers basic case, need float64, int types
    // TODO: Missing test for Reciprocal boundary conditions - Very small values (near zero), very large values, +/-Inf
    // TODO: Missing test coverage for Sqrt of negative values - No tests validate NaN behavior - Need edge case tests
    // TODO: Missing test coverage for Log of negative/zero values - No tests validate NaN/-Inf behavior - Need edge case tests
    // TODO: Missing test coverage for Pow edge cases - No tests for 0^0, negative^non-integer, Inf^0, etc. - Need edge case tests
    // TODO: Missing test coverage for Gelu approximate mode - Gelu has 'approximate' attribute but no tests - Need tests for both modes
}
