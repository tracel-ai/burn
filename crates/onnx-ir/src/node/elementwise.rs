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
//! - **Max**: Element-wise maximum (Opset 1+)
//! - **Min**: Element-wise minimum (Opset 1+)
//! - **BitwiseNot**: Bitwise NOT (Opset 18+)
//!
//!
//! ## Implementation Notes
//! - No opset validation currently performed for binary operations (see TODO at line 108)

use crate::ir::{Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
};

/// Node representation for element-wise binary operations
#[derive(Debug, Clone)]
pub struct ElementwiseBinaryNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub node_type: crate::ir::NodeType,
}

/// Node representation for element-wise unary operations
#[derive(Debug, Clone)]
pub struct ElementwiseUnaryNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub node_type: crate::ir::NodeType,
}

/// Node processor for element-wise binary
/// Node processor for element-wise unary operations
/// Used for: Neg, Abs, Ceil, Floor, Sqrt, Exp, Log, Sin, Cos, etc.
pub(crate) struct ElementwiseUnaryProcessor;

impl NodeProcessor for ElementwiseUnaryProcessor {
    type Config = ();

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
        // Note: Only operations still using ElementwiseUnaryNode are handled here
        let min_opset = match node.node_type {
            // Opset 7 operations (inverse trigonometric functions)
            crate::ir::NodeType::Acos | crate::ir::NodeType::Asin | crate::ir::NodeType::Atan => 7,
            // Opset 9 operations (inverse hyperbolic functions)
            crate::ir::NodeType::Asinh
            | crate::ir::NodeType::Acosh
            | crate::ir::NodeType::Atanh => 9,
            // Other activation functions
            crate::ir::NodeType::Elu => 6,
            crate::ir::NodeType::Selu => 6,
            crate::ir::NodeType::Celu => 12,
            crate::ir::NodeType::Mish => 18,
            crate::ir::NodeType::Softplus => 1,
            crate::ir::NodeType::Softsign => 1,
            crate::ir::NodeType::ThresholdedRelu => 10,
            crate::ir::NodeType::HardSwish => 14,
            _ => {
                return Err(ProcessError::Custom(format!(
                    "Unexpected node type for ElementwiseUnaryProcessor: {:?}",
                    node.node_type
                )));
            }
        };

        crate::processor::validate_opset(opset, min_opset)?;

        same_as_input(node);
        Ok(())
    }

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        use crate::ir::NodeType;

        let node = ElementwiseUnaryNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            node_type: builder.node_type.clone(),
        };

        match builder.node_type {
            // Inverse trig functions (still using ElementwiseUnaryNode)
            NodeType::Acos => Node::Acos(node),
            NodeType::Asin => Node::Asin(node),
            NodeType::Atan => Node::Atan(node),
            NodeType::Asinh => Node::Asinh(node),
            NodeType::Acosh => Node::Acosh(node),
            NodeType::Atanh => Node::Atanh(node),
            // Other activation functions (still using ElementwiseUnaryNode)
            NodeType::Elu => Node::Elu(node),
            NodeType::Selu => Node::Selu(node),
            NodeType::Celu => Node::Celu(node),
            NodeType::Mish => Node::Mish(node),
            NodeType::Softplus => Node::Softplus(node),
            NodeType::Softsign => Node::Softsign(node),
            NodeType::ThresholdedRelu => Node::ThresholdedRelu(node),
            NodeType::HardSwish => Node::HardSwish(node),
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
    fn test_elementwise_unary_processor() {
        let processor = ElementwiseUnaryProcessor;
        let prefs = OutputPreferences::new();

        let mut node = crate::ir::NodeBuilder {
            node_type: NodeType::Asin,
            name: "test_asin".to_string(),
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
            node_type: NodeType::ThresholdedRelu,
            name: "test_thresholded_relu".to_string(),
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

        let result = processor.infer_types(&mut node, 9, &prefs);
        assert!(matches!(
            result,
            Err(ProcessError::UnsupportedOpset {
                required: 10,
                actual: 9
            })
        ));
    }
}
