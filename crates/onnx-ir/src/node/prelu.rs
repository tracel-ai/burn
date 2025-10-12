//! Processor for PRelu operation

use crate::ir::Node;
use crate::processor::NodeProcessor;
use crate::util::same_as_input_broadcast;

/// Node processor for PRelu (Parametric ReLU) operation
/// PRelu applies: f(x) = x if x >= 0 else slope * x
/// where slope is learned and can be broadcasted
pub struct PReluProcessor;

impl NodeProcessor for PReluProcessor {
    fn first_pass(&self, node: &mut Node, _opset: usize) {
        same_as_input_broadcast(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, ElementType, NodeType, TensorType};

    #[test]
    fn test_prelu_processor() {
        let processor = PReluProcessor;

        let mut node = crate::ir::Node {
            node_type: NodeType::PRelu,
            name: "test_prelu".to_string(),
            inputs: vec![
                Argument {
                    name: "x".to_string(),
                    ty: ArgType::Tensor(TensorType {
                        elem_type: ElementType::Float32,
                        rank: 3,
                        static_shape: None,
                    }),
                    value_store: None,
                },
                Argument {
                    name: "slope".to_string(),
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
