//! Processors for element-wise operations (Add, Sub, Mul, Div, etc.)

use crate::ir::Node;
use crate::processor::{NodeProcessor, ProcessorContext};
use crate::util::same_as_input_broadcast;

/// Node processor for element-wise binary operations that support broadcasting
/// Used for: Add, Sub, Mul, Div, Mod, Pow, etc.
pub struct ElementwiseBinaryProcessor;

impl NodeProcessor for ElementwiseBinaryProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (7, None) // Most element-wise ops use opset 7+ for broadcasting
    }

    fn infer_outputs(&self, node: &mut Node, _context: &ProcessorContext) {
        same_as_input_broadcast(node);
    }
}

/// Node processor for element-wise unary operations
/// Used for: Neg, Abs, Ceil, Floor, Sqrt, Exp, Log, Sin, Cos, etc.
pub struct ElementwiseUnaryProcessor;

impl NodeProcessor for ElementwiseUnaryProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (6, None) // Unary ops generally stable from opset 6+
    }

    fn infer_outputs(&self, node: &mut Node, _context: &ProcessorContext) {
        crate::util::same_as_input(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, NodeType, TensorType};

    #[test]
    fn test_elementwise_binary_processor() {
        let processor = ElementwiseBinaryProcessor;
        assert_eq!(processor.supported_opset_range(), (7, None));

        let mut node = crate::ir::Node {
            node_type: NodeType::Add,
            name: "test_add".to_string(),
            inputs: vec![
                Argument {
                    name: "a".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 2,
                        static_shape: None,
                    }),
                    value: None,
                    passed: true,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 2,
                        static_shape: None,
                    }),
                    value: None,
                    passed: true,
                },
            ],
            outputs: vec![Argument {
                name: "c".to_string(),
                ty: ArgType::default(),
                value: None,
                passed: false,
            }],
            attrs: Default::default(),
        };

        let ctx = ProcessorContext::new(16);
        processor.infer_outputs(&mut node, &ctx);

        // Output should be rank 2
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 2),
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_elementwise_unary_processor() {
        let processor = ElementwiseUnaryProcessor;
        assert_eq!(processor.supported_opset_range(), (6, None));

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
                value: None,
                passed: true,
            }],
            outputs: vec![Argument {
                name: "b".to_string(),
                ty: ArgType::default(),
                value: None,
                passed: false,
            }],
            attrs: Default::default(),
        };

        let ctx = ProcessorContext::new(16);
        processor.infer_outputs(&mut node, &ctx);

        // Output should match input
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 3);
                assert_eq!(t.elem_type, ElementType::Float32);
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
