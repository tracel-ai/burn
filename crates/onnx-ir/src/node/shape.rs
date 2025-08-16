use crate::ir::{ArgType, Node};

pub fn shape_config(curr: &Node) -> (usize, usize) {
    if curr.inputs.len() != 1 {
        panic!(
            "Shape: multiple inputs are not supported (got {:?})",
            curr.inputs.len()
        );
    }

    // Extract the rank/dimension count from the input
    let rank = match &curr.inputs.first().unwrap().ty {
        ArgType::Tensor(tensor) => tensor.rank,
        ArgType::Shape(rank) => {
            // When Shape is applied to a Shape type, we're getting the "shape of the shape"
            // which is just the number of dimensions (rank) as a 1D tensor
            *rank
        }
        _ => panic!(
            "Shape operation expects Tensor or Shape input, got {:?}",
            curr.inputs.first().unwrap().ty
        ),
    };

    // Default: all axes up to the last one (included)
    let mut start_dim: i64 = 0;
    let mut end_dim: i64 = rank as i64;

    // Extract the attributes
    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "start" => start_dim = value.clone().into_i64(),
            "end" => end_dim = value.clone().into_i64(),
            _ => {}
        }
    }

    // If dim is negative, it is counted from the end
    if start_dim < 0 {
        start_dim += rank as i64;
    }
    if end_dim < 0 {
        end_dim += rank as i64;
    }

    (start_dim as usize, end_dim as usize)
}

/// Update output type for Shape operation (rank 1).
pub fn shape_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("Shape: multiple inputs are not supported: {node:?}");
    }

    // Special case: Shape of Shape returns a 1D tensor with single element (the rank)
    if let ArgType::Shape(rank) = &node.inputs[0].ty {
        // The shape of a shape is always a 1D tensor with one element
        // containing the rank/number of dimensions
        // Since Shape types are [i64; N], getting their shape gives us [N] which is Shape(1)
        log::debug!(
            "Shape operation on Shape({}) input for node {}: output is Shape(1)",
            rank,
            node.name
        );
        node.outputs[0].ty = ArgType::Shape(1);
        return;
    }

    let (start, end) = shape_config(node);
    let dim = end - start;
    log::debug!(
        "Shape operation for node {}: start={}, end={}, dim={}",
        node.name,
        start,
        end,
        dim
    );
    node.outputs[0].ty = ArgType::Shape(dim);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(start: Option<i64>, end: Option<i64>, rank: usize) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Shape, "test_shape")
            .input_tensor_f32("data", rank, None)
            .output_tensor_i64("shape", 1, None);

        if let Some(start_val) = start {
            builder = builder.attr_int("start", start_val);
        }

        if let Some(end_val) = end {
            builder = builder.attr_int("end", end_val);
        }

        builder.build()
    }

    #[test]
    fn test_shape_config_defaults() {
        let node = create_test_node(None, None, 4);
        let (start, end) = shape_config(&node);
        assert_eq!(start, 0);
        assert_eq!(end, 4);
    }

    #[test]
    fn test_shape_config_with_start() {
        let node = create_test_node(Some(1), None, 4);
        let (start, end) = shape_config(&node);
        assert_eq!(start, 1);
        assert_eq!(end, 4);
    }

    #[test]
    fn test_shape_config_with_end() {
        let node = create_test_node(None, Some(3), 4);
        let (start, end) = shape_config(&node);
        assert_eq!(start, 0);
        assert_eq!(end, 3);
    }

    #[test]
    fn test_shape_config_with_start_and_end() {
        let node = create_test_node(Some(1), Some(3), 4);
        let (start, end) = shape_config(&node);
        assert_eq!(start, 1);
        assert_eq!(end, 3);
    }

    #[test]
    fn test_shape_config_negative_dims() {
        let node = create_test_node(Some(-2), Some(-1), 4);
        let (start, end) = shape_config(&node);
        assert_eq!(start, 2); // -2 + 4 = 2
        assert_eq!(end, 3); // -1 + 4 = 3
    }

    #[test]
    #[should_panic(expected = "Shape: multiple inputs are not supported")]
    fn test_shape_config_multiple_inputs() {
        let mut node = create_test_node(None, None, 4);
        // Add an extra input to cause the expected panic
        node.inputs.push(crate::ir::Argument {
            name: "extra".to_string(),
            ty: crate::ir::ArgType::Tensor(crate::ir::TensorType {
                elem_type: crate::ir::ElementType::Float32,
                rank: 4,
                static_shape: None,
            }),
            value: None,
            passed: true,
        });
        let _ = shape_config(&node);
    }

    #[test]
    fn test_shape_of_shape() {
        // Test Shape operation on Shape input
        let mut node = NodeBuilder::new(NodeType::Shape, "test_shape_of_shape")
            .add_input("shape_input", ArgType::Shape(3)) // Input is Shape(3) - a 3D shape
            .output_tensor_i64("output", 1, None)
            .build();

        // Before update
        assert!(matches!(node.inputs[0].ty, ArgType::Shape(3)));

        // Apply shape_update_outputs
        shape_update_outputs(&mut node);

        // After update: Shape of Shape(3) should give Shape(1)
        // because [i64; 3] has shape [3] which is 1D
        assert!(matches!(node.outputs[0].ty, ArgType::Shape(1)));
    }

    #[test]
    fn test_shape_config_with_shape_input() {
        // Test shape_config with Shape input
        let node = NodeBuilder::new(NodeType::Shape, "test_shape_config_shape")
            .add_input("shape_input", ArgType::Shape(5)) // Input is Shape(5)
            .output_tensor_i64("output", 1, None)
            .build();

        let (start, end) = shape_config(&node);
        // Shape(5) means a 5-dimensional shape, so getting its shape
        // would be from 0 to 5 (the full extent)
        assert_eq!(start, 0);
        assert_eq!(end, 5);
    }

    #[test]
    fn test_shape_of_shape_with_attributes() {
        // Test Shape operation on Shape input with start/end attributes
        let mut node = NodeBuilder::new(NodeType::Shape, "test_shape_of_shape_attrs")
            .add_input("shape_input", ArgType::Shape(4)) // Input is Shape(4)
            .output_tensor_i64("output", 1, None)
            .attr_int("start", 1)
            .attr_int("end", 3)
            .build();

        shape_update_outputs(&mut node);

        // Even with start/end attributes, Shape of Shape always outputs Shape(1)
        // because we're getting the shape of the shape array itself
        assert!(matches!(node.outputs[0].ty, ArgType::Shape(1)));
    }
}
