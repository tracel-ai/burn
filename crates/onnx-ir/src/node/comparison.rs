use crate::ir::{ArgType, ElementType, Node, TensorType};

/// Update output type for comparison operations (e.g., Equal, Greater) to max input rank.
pub fn elementwise_comparison_outputs(node: &mut Node) {
    log::debug!("Elementwise comparison for node {}", node.name);

    let max_rank = node.inputs.iter().fold(0, |acc, input| match &input.ty {
        ArgType::Tensor(tensor) => acc.max(tensor.rank),
        ArgType::Scalar(_) => acc,
        _ => panic!("Invalid input type for comparison op"),
    });

    log::debug!("Max rank for comparison node {}: {}", node.name, max_rank);

    if max_rank == 0 {
        node.outputs[0].ty = ArgType::Scalar(ElementType::Bool);
        log::debug!("Scalar boolean result for node {}", node.name);
    } else {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: ElementType::Bool,
            rank: max_rank,
            static_shape: None,
        });
        log::debug!(
            "Tensor boolean result for node {} with rank {}",
            node.name,
            max_rank
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(input1_rank: usize, input2_rank: usize) -> Node {
        NodeBuilder::new(NodeType::Equal, "test_comparison")
            .input_tensor_f32("A", input1_rank, None)
            .input_tensor_f32("B", input2_rank, None)
            .output_tensor_bool("result", 0, None) // rank will be updated
            .build()
    }

    #[test]
    fn test_comparison_rank_broadcasting() {
        let mut node = create_test_node(2, 3);
        elementwise_comparison_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Bool);
                assert_eq!(tensor.rank, 3); // max(2, 3) = 3
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_comparison_scalar_result() {
        let mut node = create_test_node(0, 0);

        // Convert inputs to scalars
        node.inputs[0].ty = ArgType::Scalar(ElementType::Float32);
        node.inputs[1].ty = ArgType::Scalar(ElementType::Float32);

        elementwise_comparison_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Bool);
            }
            _ => panic!("Expected scalar output"),
        }
    }

    #[test]
    #[should_panic(expected = "Invalid input type for comparison op")]
    fn test_comparison_invalid_input() {
        let mut node = create_test_node(2, 2);
        node.inputs[0].ty = ArgType::Shape(2);
        elementwise_comparison_outputs(&mut node);
    }
}
