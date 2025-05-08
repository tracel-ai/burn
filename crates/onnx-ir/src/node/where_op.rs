use crate::ir::{ArgType, ElementType, Node, TensorType};
use core::cmp::max;

/// Update output rank for Where to max input rank.
pub fn where_update_outputs(node: &mut Node) {
    log::debug!("Where rank inference for node {}", node.name);

    let condition = &node.inputs[0].ty;
    let x = &node.inputs[1].ty;
    let y = &node.inputs[2].ty;
    let elem_type = x.elem_type().clone();
    assert_eq!(
        *condition.elem_type(),
        ElementType::Bool,
        "Where condition must be boolean!"
    );
    assert_eq!(
        elem_type,
        *y.elem_type(),
        "Where x and y have different element types!"
    );

    log::debug!(
        "Where input ranks for {}: condition={}, x={}, y={}",
        node.name,
        condition.rank(),
        x.rank(),
        y.rank()
    );

    let output_rank = max(condition.rank(), max(x.rank(), y.rank()));
    log::debug!("Where output rank for {}: {}", node.name, output_rank);

    if output_rank == 0 {
        node.outputs[0].ty = ArgType::Scalar(elem_type);
        log::debug!("Where result for {} is scalar", node.name);
    } else {
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type,
            rank: output_rank,
            static_shape: None,
        });
        log::debug!(
            "Where result for {} is tensor with rank {}",
            node.name,
            output_rank
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{NodeType, TensorType};
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(condition_rank: usize, x_rank: usize, y_rank: usize) -> Node {
        NodeBuilder::new(NodeType::Where, "test_where")
            .input_tensor_bool("condition", condition_rank, None)
            .input_tensor_f32("X", x_rank, None)
            .input_tensor_f32("Y", y_rank, None)
            .output_tensor_f32("output", 0, None) // Rank will be updated
            .build()
    }

    #[test]
    fn test_where_basic() {
        let mut node = create_test_node(2, 3, 2);
        where_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3); // max(2, max(3, 2)) = 3
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_where_scalar_result() {
        let mut node = create_test_node(0, 0, 0);
        where_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Float32);
            }
            _ => panic!("Expected scalar output"),
        }
    }

    #[test]
    #[should_panic(expected = "Where condition must be boolean!")]
    fn test_where_invalid_condition() {
        let mut node = create_test_node(2, 2, 2);
        node.inputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: ElementType::Float32, // Not boolean
            rank: 2,
            static_shape: None,
        });
        where_update_outputs(&mut node);
    }

    #[test]
    #[should_panic(expected = "Where x and y have different element types!")]
    fn test_where_mismatched_types() {
        let mut node = create_test_node(2, 2, 2);
        node.inputs[2].ty = ArgType::Tensor(TensorType {
            elem_type: ElementType::Int64, // Different from X
            rank: 2,
            static_shape: None,
        });
        where_update_outputs(&mut node);
    }
}
