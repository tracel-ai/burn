use crate::ir::{ArgType, Node};

pub fn shape_config(curr: &Node) -> (usize, usize) {
    if curr.inputs.len() != 1 {
        panic!(
            "Shape: multiple inputs are not supported (got {:?})",
            curr.inputs.len()
        );
    }

    // Extract the shape of the input tensor
    let tensor = match curr.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Default: all axes up to the last one (included)
    let mut start_dim: i64 = 0;
    let mut end_dim: i64 = tensor.rank as i64;

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
        start_dim += tensor.rank as i64;
    }
    if end_dim < 0 {
        end_dim += tensor.rank as i64;
    }

    (start_dim as usize, end_dim as usize)
}

/// Update output type for Shape operation (rank 1).
pub fn shape_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("Shape: multiple inputs are not supported: {:?}", node);
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
        });
        let _ = shape_config(&node);
    }
}
