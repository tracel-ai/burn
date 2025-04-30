use crate::ir::{ArgType, Node, TensorType};

/// Update output rank for Split (same as input).
pub fn split_update_outputs(node: &mut Node) {
    log::debug!("Split rank inference for node {}", node.name);

    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Split: Input must be a tensor"),
    };
    log::debug!("Split input rank for {}: {}", node.name, tensor.rank);
    log::debug!(
        "Split will generate {} outputs for {}",
        node.outputs.len(),
        node.name
    );

    for (i, output_arg) in node.outputs.iter_mut().enumerate() {
        output_arg.ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: tensor.rank,
            static_shape: None,
        });
        log::debug!("Split output {} rank for {}: {}", i, node.name, tensor.rank);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, ElementType, NodeType};
    use std::collections::HashMap;

    fn create_test_node(input_rank: usize, num_outputs: usize) -> Node {
        let inputs = vec![Argument {
            name: "input".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: input_rank,
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        let mut outputs = Vec::new();
        for i in 0..num_outputs {
            outputs.push(Argument {
                name: format!("output_{}", i),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 0, // Will be updated
                    static_shape: None,
                }),
                value: None,
                passed: true,
            });
        }

        let attrs = HashMap::new();

        Node {
            node_type: NodeType::Split,
            name: "test_split".to_string(),
            inputs,
            outputs,
            attrs,
        }
    }

    #[test]
    fn test_split_single_output() {
        let mut node = create_test_node(3, 1);
        split_update_outputs(&mut node);

        assert_eq!(node.outputs.len(), 1);
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_split_multiple_outputs() {
        let mut node = create_test_node(4, 3);
        split_update_outputs(&mut node);

        assert_eq!(node.outputs.len(), 3);
        for output in &node.outputs {
            match &output.ty {
                ArgType::Tensor(tensor) => {
                    assert_eq!(tensor.elem_type, ElementType::Float32);
                    assert_eq!(tensor.rank, 4);
                }
                _ => panic!("Expected tensor output"),
            }
        }
    }

    #[test]
    #[should_panic(expected = "Split: Input must be a tensor")]
    fn test_split_invalid_input() {
        let mut node = create_test_node(3, 2);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        split_update_outputs(&mut node);
    }
}
