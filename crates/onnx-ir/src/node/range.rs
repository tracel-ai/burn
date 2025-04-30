use crate::ir::{ArgType, ElementType, Node, TensorType};

/// Update output rank for Range (always rank 1).
pub fn range_update_outputs(node: &mut Node) {
    log::debug!("Range rank inference for node {}", node.name);

    if node.inputs.len() != 3 {
        panic!("Range: expected 3 inputs, found {}", node.inputs.len());
    }
    log::debug!(
        "Range operation always produces rank 1 tensor for {}",
        node.name
    );

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: ElementType::Int64,
        rank: 1,
        static_shape: None,
    });

    log::debug!("Range output rank for {}: 1", node.name);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, NodeType};
    use std::collections::HashMap;

    fn create_test_node() -> Node {
        let inputs = vec![
            Argument {
                name: "start".to_string(),
                ty: ArgType::Scalar(ElementType::Int64),
                value: None,
                passed: true,
            },
            Argument {
                name: "limit".to_string(),
                ty: ArgType::Scalar(ElementType::Int64),
                value: None,
                passed: true,
            },
            Argument {
                name: "delta".to_string(),
                ty: ArgType::Scalar(ElementType::Int64),
                value: None,
                passed: true,
            },
        ];

        let outputs = vec![Argument {
            name: "output".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: 0, // Will be updated
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        Node {
            node_type: NodeType::Range,
            name: "test_range".to_string(),
            inputs,
            outputs,
            attrs: HashMap::new(),
        }
    }

    #[test]
    fn test_range_output() {
        let mut node = create_test_node();
        range_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    #[should_panic(expected = "Range: expected 3 inputs, found 2")]
    fn test_range_missing_inputs() {
        let mut node = create_test_node();
        node.inputs.pop();
        range_update_outputs(&mut node);
    }
}
