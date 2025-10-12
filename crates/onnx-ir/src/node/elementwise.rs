//! Processors for element-wise operations

use crate::ir::Node;
use crate::processor::NodeProcessor;

/// Node processor for element-wise unary operations
/// Used for: Neg, Abs, Ceil, Floor, Sqrt, Exp, Log, Sin, Cos, etc.
pub struct ElementwiseUnaryProcessor;

impl NodeProcessor for ElementwiseUnaryProcessor {
    fn first_pass(&self, node: &mut Node, _opset: usize) {
        crate::util::same_as_input(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, NodeType, TensorType};

    #[test]
    fn test_elementwise_unary_processor() {
        let processor = ElementwiseUnaryProcessor;

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
                value_store: None,
            }],
            outputs: vec![Argument {
                name: "b".to_string(),
                ty: ArgType::default(),
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.first_pass(&mut node, 16);

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
