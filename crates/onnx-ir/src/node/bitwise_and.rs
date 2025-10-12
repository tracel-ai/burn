//! Processor for BitwiseAnd operation

use crate::ir::Node;
use crate::processor::NodeProcessor;
use crate::util::same_as_input_broadcast;

/// Node processor for BitwiseAnd operation
pub struct BitwiseAndProcessor;

impl NodeProcessor for BitwiseAndProcessor {
    fn first_pass(&self, node: &mut Node, _opset: usize) {
        same_as_input_broadcast(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, NodeType, TensorType};

    #[test]
    fn test_bitwise_and_processor() {
        let processor = BitwiseAndProcessor;

        let mut node = crate::ir::Node {
            node_type: NodeType::BitwiseAnd,
            name: "test_bitwise_and".to_string(),
            inputs: vec![
                Argument {
                    name: "a".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Int32,
                        rank: 2,
                        static_shape: None,
                    }),
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Int32,
                        rank: 2,
                        static_shape: None,
                    }),
                    value_store: None,
                },
            ],
            outputs: vec![Argument {
                name: "result".to_string(),
                ty: ArgType::default(),
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.first_pass(&mut node, 16);

        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 2),
            _ => panic!("Expected tensor output"),
        }
    }
}
