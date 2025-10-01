use crate::{
    ir::{ArgType, ElementType, Node, TensorType},
    util::{compute_broadcast_rank, compute_broadcast_static_shape},
};

/// Get element type from ArgType, handling Shape types specially
fn get_elem_type(arg_type: &ArgType) -> ElementType {
    match arg_type {
        ArgType::Scalar(elem_type) => elem_type.clone(),
        ArgType::Tensor(tensor) => tensor.elem_type.clone(),
        ArgType::Shape(_) => ElementType::Int64, // Shape types are always i64
    }
}

/// Check if output should be a Shape type
fn should_output_shape(
    x: &ArgType,
    y: &ArgType,
    output_rank: usize,
    elem_type: &ElementType,
) -> bool {
    // Output Shape if both inputs are Shape and output would be 1D int64
    matches!(x, ArgType::Shape(_))
        && matches!(y, ArgType::Shape(_))
        && output_rank == 1
        && *elem_type == ElementType::Int64
}

/// Get size of Shape type, or 1 for other types
fn get_shape_size(arg_type: &ArgType) -> usize {
    match arg_type {
        ArgType::Shape(size) => *size,
        _ => 1,
    }
}

/// Update output type for Where operation.
///
/// The Where operation selects elements from x or y based on condition.
/// Output shape is the broadcasted shape of all three inputs.
/// Output element type is taken from x and y (which must match).
pub fn where_update_outputs(node: &mut Node) {
    log::debug!("Where rank inference for node {}", node.name);

    let condition = &node.inputs[0].ty;
    let x = &node.inputs[1].ty;
    let y = &node.inputs[2].ty;

    // Get element types, handling Shape types specially
    let x_elem_type = get_elem_type(x);
    let y_elem_type = get_elem_type(y);
    let condition_elem_type = get_elem_type(condition);

    if !matches!(condition, ArgType::Shape(_)) {
        assert_eq!(
            condition_elem_type,
            ElementType::Bool,
            "Where condition must be boolean!"
        );
    }

    let elem_type = if x_elem_type == y_elem_type {
        x_elem_type
    } else if matches!(x, ArgType::Shape(_)) {
        y_elem_type
    } else if matches!(y, ArgType::Shape(_)) {
        x_elem_type
    } else {
        panic!(
            "Where x and y have different element types! ({:?} vs {:?})",
            x_elem_type, y_elem_type
        );
    };

    log::debug!(
        "Where input ranks for {}: condition={}, x={}, y={}",
        node.name,
        condition.rank(),
        x.rank(),
        y.rank()
    );

    let output_rank = compute_broadcast_rank(&node.inputs);
    log::debug!("Where output rank for {}: {}", node.name, output_rank);

    // Determine output type
    if output_rank == 0 {
        node.outputs[0].ty = ArgType::Scalar(elem_type);
        log::debug!("Where result for {} is scalar", node.name);
    } else if should_output_shape(x, y, output_rank, &elem_type) {
        // If both inputs are Shape types and output is 1D int64, preserve Shape type
        let shape_size = get_shape_size(x).max(get_shape_size(y));
        node.outputs[0].ty = ArgType::Shape(shape_size);
        log::debug!(
            "Where result for {} is Shape({}) type",
            node.name,
            shape_size
        );
    } else {
        // Try to propagate static shape using the shared broadcast helper
        let static_shape = compute_broadcast_static_shape(&node.inputs);

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type,
            rank: output_rank,
            static_shape,
        });
        log::debug!(
            "Where result for {} is tensor with rank {}, static_shape: {:?}",
            node.name,
            output_rank,
            node.outputs[0].ty.static_shape()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
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

        // Replace condition with non-boolean tensor
        let non_bool_input = NodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_f32("x", 2, None)
            .build()
            .inputs
            .pop()
            .unwrap();

        node.inputs[0] = non_bool_input;
        where_update_outputs(&mut node);
    }

    #[test]
    #[should_panic(expected = "Where x and y have different element types!")]
    fn test_where_mismatched_types() {
        let mut node = create_test_node(2, 2, 2);

        // Replace Y with int64 tensor (different from X's float32)
        let int64_input = NodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_i64("y", 2, None)
            .build()
            .inputs
            .pop()
            .unwrap();

        node.inputs[2] = int64_input;
        where_update_outputs(&mut node);
    }

    #[test]
    fn test_where_with_shape_inputs() {
        let mut node = create_test_node(1, 0, 0);

        // Replace X and Y with Shape types
        node.inputs[1].ty = ArgType::Shape(3);
        node.inputs[2].ty = ArgType::Shape(3);

        where_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Shape(size) => {
                assert_eq!(*size, 3); // Should preserve Shape type
            }
            _ => panic!("Expected Shape output"),
        }
    }

    #[test]
    fn test_where_static_shape_propagation() {
        // Test that static shapes propagate correctly through Where
        let mut node = NodeBuilder::new(NodeType::Where, "test_where")
            .input_tensor_bool("condition", 2, Some(vec![2, 2]))
            .input_tensor_f32("X", 2, Some(vec![2, 2]))
            .input_tensor_f32("Y", 2, Some(vec![2, 2]))
            .output_tensor_f32("output", 0, None)
            .build();

        where_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, Some(vec![2, 2]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_where_static_shape_propagation_partial() {
        // Test that static shape propagates even when only some inputs have it
        let mut node = NodeBuilder::new(NodeType::Where, "test_where")
            .input_tensor_bool("condition", 2, None) // No static shape
            .input_tensor_f32("X", 2, Some(vec![3, 4])) // Has static shape
            .input_tensor_f32("Y", 2, Some(vec![3, 4])) // Has static shape
            .output_tensor_f32("output", 0, None)
            .build();

        where_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 2);
                // Since X and Y have the same static shape, it should propagate
                assert_eq!(tensor.static_shape, Some(vec![3, 4]));
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
