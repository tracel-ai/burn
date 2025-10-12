//! Processor for Add operation

use crate::ir::Node;
use crate::processor::NodeProcessor;
use crate::util::same_as_input_broadcast;

/// Node processor for Add operation
pub struct AddProcessor;

impl NodeProcessor for AddProcessor {
    fn first_pass(&self, node: &mut Node, _opset: usize) {
        // make sure there are two only inputs
        assert_eq!(node.inputs.len(), 2);

        // if the first input is Shape and the second is Constant
        // then make the second arg Should be Shape
        if node.inputs[0].ty.is_shape() && node.inputs[1].has_value() {
            node.inputs[1].should_be(node.inputs[0].ty.clone());
        }

        // if the first input is Constant and the second is Shape
        // then make the first arg Should be Shape
        if node.inputs[1].ty.is_shape() && node.inputs[0].has_value() {
            node.inputs[0].should_be(node.inputs[1].ty.clone());
        }

        same_as_input_broadcast(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, NodeType, TensorType};

    #[test]
    fn test_add_processor() {
        let processor = AddProcessor;

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
                name: "c".to_string(),
                ty: ArgType::default(),
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.first_pass(&mut node, 16);

        // Output should be rank 2
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => assert_eq!(t.rank, 2),
            _ => panic!("Expected tensor output"),
        }
    }
}
