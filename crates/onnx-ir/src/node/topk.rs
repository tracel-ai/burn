use crate::ir::{ArgType, ElementType, Node, TensorType};

/// Update output rank for TopK (same as input rank).
pub fn top_k_update_output(node: &mut Node) {
    log::debug!("TopK rank inference for node {}", node.name);

    let rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("TopK: invalid input type"),
    };
    log::debug!("TopK input rank for {}: {}", node.name, rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: node.inputs[0].ty.elem_type().clone(),
        rank,
        static_shape: None,
    });
    node.outputs[1].ty = ArgType::Tensor(TensorType {
        elem_type: ElementType::Int64,
        rank,
        static_shape: None,
    });

    log::debug!(
        "TopK output rank for {}: {} (both outputs)",
        node.name,
        rank
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, NodeType};
    use std::collections::HashMap;

    fn create_test_node(input_rank: usize) -> Node {
        let inputs = vec![
            Argument {
                name: "X".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: input_rank,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
            Argument {
                name: "K".to_string(),
                ty: ArgType::Scalar(ElementType::Int64),
                value: None,
                passed: true,
            },
        ];

        let outputs = vec![
            Argument {
                name: "Values".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 0, // Will be updated
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
            Argument {
                name: "Indices".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: 0, // Will be updated
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
        ];

        let attrs = HashMap::new();

        Node {
            node_type: NodeType::TopK,
            name: "test_topk".to_string(),
            inputs,
            outputs,
            attrs,
        }
    }

    #[test]
    fn test_topk_basic() {
        let mut node = create_test_node(3);
        top_k_update_output(&mut node);

        assert_eq!(node.outputs.len(), 2);

        // Check first output (values)
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output for values"),
        }

        // Check second output (indices)
        match &node.outputs[1].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output for indices"),
        }
    }

    #[test]
    #[should_panic(expected = "TopK: invalid input type")]
    fn test_topk_invalid_input() {
        let mut node = create_test_node(3);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        top_k_update_output(&mut node);
    }
}
