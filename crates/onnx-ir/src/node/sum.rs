//! Processor for Sum operation

use crate::ir::Node;
use crate::processor::NodeProcessor;
use crate::util::{same_as_input_broadcast, validate_opset};

/// Node processor for Sum operation
/// Note: Sum is variadic (can take multiple inputs), not strictly binary
pub struct SumProcessor;

impl NodeProcessor for SumProcessor {
    fn first_pass(&self, node: &mut Node, opset: usize) {
        // Sum implementation supports opset 8+ (numpy broadcasting)
        validate_opset(&node.node_type, opset, 8);

        same_as_input_broadcast(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, NodeType, TensorType};

    #[test]
    fn test_sum_processor_two_inputs() {
        let processor = SumProcessor;

        let mut node = crate::ir::Node {
            node_type: NodeType::Sum,
            name: "test_sum".to_string(),
            inputs: vec![
                Argument {
                    name: "a".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 2,
                        static_shape: None,
                    }),
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
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

    #[test]
    fn test_sum_processor_multiple_inputs() {
        let processor = SumProcessor;

        let mut node = crate::ir::Node {
            node_type: NodeType::Sum,
            name: "test_sum".to_string(),
            inputs: vec![
                Argument {
                    name: "a".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 3,
                        static_shape: None,
                    }),
                    value_store: None,
                },
                Argument {
                    name: "b".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 3,
                        static_shape: None,
                    }),
                    value_store: None,
                },
                Argument {
                    name: "c".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 3,
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
            ArgType::Tensor(t) => assert_eq!(t.rank, 3),
            _ => panic!("Expected tensor output"),
        }
    }
}
