use crate::ir::{ArgType, ElementType, Node, TensorType};
use core::cmp::max;

/// Update output rank and type for MatMulInteger based on input ranks.
pub fn matmulinteger_update_outputs(node: &mut Node) {
    match (&node.inputs[0].ty, &node.inputs[1].ty) {
        (ArgType::Tensor(a), ArgType::Tensor(b)) => {
            let mut out_rank = max(a.rank, b.rank);

            // Special cases: vector–matrix or matrix–vector reduces rank by 1
            if (a.rank >= 2 && b.rank == 1) || (a.rank == 1 && b.rank >= 2) {
                out_rank -= 1;
            }

            // ONNX spec: output is always int32
            // ONNX spec: MatMulInteger output is always int32
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Int32,
                rank: out_rank,
                static_shape: None, // or Some(...) if you’ve inferred it
            });
        }
        _ => panic!("MatMulInteger expects tensor inputs"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ElementType, NodeType};
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(a_rank: usize, b_rank: usize) -> Node {
        NodeBuilder::new(NodeType::MatMulInteger, "test_matmulinteger")
            .input_tensor_i32("A", a_rank, None)
            .input_tensor_i32("B", b_rank, None)
            .output_tensor_i32("Y", 0, None) // rank will be updated
            .build()
    }

    #[test]
    fn test_update_outputs_standard_case() {
        let mut node = create_test_node(2, 2);
        matmulinteger_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_update_outputs_vector_matrix() {
        let mut node = create_test_node(1, 2);
        matmulinteger_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int32);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    #[should_panic(expected = "MatMulInteger expects tensor inputs")]
    fn test_invalid_input() {
        let mut node = create_test_node(2, 2);
        node.inputs[0].ty = ArgType::Scalar(ElementType::Int32);
        matmulinteger_update_outputs(&mut node);
    }
}
#[cfg(test)]
mod tests2 {
    use super::*;
    use crate::ir::{ElementType, NodeType};
    use crate::node::test_utils::NodeBuilder;

    fn mk(a_rank: usize, b_rank: usize) -> Node {
        NodeBuilder::new(NodeType::MatMulInteger, "mmint")
            .input_tensor_i32("A", a_rank, None)
            .input_tensor_i32("B", b_rank, None)
            .output_tensor_i32("Y", 0, None)
            .build()
    }

    #[test]
    fn out_rank_2x2_is_2() {
        let mut n = mk(2, 2);
        matmulinteger_update_outputs(&mut n);
        match &n.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.elem_type, ElementType::Int32);
                assert_eq!(t.rank, 2);
            }
            _ => panic!("tensor expected"),
        }
    }

    #[test]
    fn vector_matrix_is_rank1() {
        let mut n = mk(1, 2);
        matmulinteger_update_outputs(&mut n);
        match &n.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.elem_type, ElementType::Int32);
                assert_eq!(t.rank, 1);
            }
            _ => panic!("tensor expected"),
        }
    }
}
