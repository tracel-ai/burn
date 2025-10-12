//! Processor for Mul operation

use crate::ir::Node;
use crate::processor::NodeProcessor;
use crate::util::same_as_input_broadcast;

/// Node processor for Mul operation
pub struct MulProcessor;

impl NodeProcessor for MulProcessor {
    fn first_pass(&self, node: &mut Node, _opset: usize) {
        same_as_input_broadcast(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, NodeType, TensorType};

    #[test]
    fn test_mul_processor() {
        let processor = MulProcessor;

        let mut node = crate::ir::Node {
            node_type: NodeType::Mul,
            name: "test_mul".to_string(),
            inputs: vec![
                Argument {
                    name: "a".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 4,
                        static_shape: None,
                    }),
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 4,
                        static_shape: None,
                    }),
                    value_store: None,
                },
            ],
            outputs: vec![Argument {
                name: "c".to_string(),
                ty: ArgType::default(),
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.first_pass(&mut node, 16);

        // Output should be rank 4
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 4),
            _ => panic!("Expected tensor output"),
        }
    }
}
