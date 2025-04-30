use crate::ir::{ArgType, Node, TensorType};
use core::cmp::max;

/// Update output rank for MatMul based on input ranks.
pub fn matmul_update_outputs(node: &mut Node) {
    log::debug!("MatMul rank inference for node {}", node.name);

    match (&node.inputs[0].ty, &node.inputs[1].ty) {
        (ArgType::Tensor(a), ArgType::Tensor(b)) => {
            log::debug!(
                "MatMul input ranks for {}: a.rank={}, b.rank={}",
                node.name,
                a.rank,
                b.rank
            );

            let mut out_rank = max(a.rank, b.rank);
            if (a.rank >= 2 && b.rank == 1) || (a.rank == 1 && b.rank >= 2) {
                out_rank -= 1;
                log::debug!(
                    "MatMul special case for node {}: reducing output rank",
                    node.name
                );
            }

            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: a.elem_type.clone(),
                rank: out_rank,
                static_shape: None,
            });

            log::debug!("MatMul output rank for {}: {}", node.name, out_rank);
        }
        _ => panic!("Only tensor inputs are valid"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, ElementType, NodeType};
    use std::collections::HashMap;

    fn create_test_node(a_rank: usize, b_rank: usize) -> Node {
        let inputs = vec![
            Argument {
                name: "A".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: a_rank,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
            Argument {
                name: "B".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: b_rank,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
        ];

        let outputs = vec![Argument {
            name: "C".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 0, // Will be updated
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        Node {
            node_type: NodeType::MatMul,
            name: "test_matmul".to_string(),
            inputs,
            outputs,
            attrs: HashMap::new(),
        }
    }

    #[test]
    fn test_matmul_standard_case() {
        let mut node = create_test_node(2, 2);
        matmul_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_matmul_broadcasting() {
        let mut node = create_test_node(3, 2);
        matmul_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_matmul_vector_matrix() {
        // When multiplying a vector (rank 1) by a matrix (rank 2)
        // the result should have rank 1 (vector)
        let mut node = create_test_node(1, 2);
        matmul_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    #[should_panic(expected = "Only tensor inputs are valid")]
    fn test_matmul_invalid_input() {
        let mut node = create_test_node(2, 2);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        matmul_update_outputs(&mut node);
    }
}
