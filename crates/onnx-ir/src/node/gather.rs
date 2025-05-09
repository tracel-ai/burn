use crate::ir::{ArgType, ElementType, Node, TensorType};

/// Update output rank for Gather based on input and indices ranks.
pub fn gather_update_outputs(node: &mut Node) {
    log::debug!("Gather rank inference for node {}", node.name);

    if node.inputs.len() != 2 {
        panic!("Gather requires two inputs: data and indices");
    }

    let indices_rank = match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        ArgType::Scalar(_) => 0,
        _ => panic!("Only tensor indices is valid, got {:?}", node.inputs[1].ty),
    };
    log::debug!("Gather indices rank for {}: {}", node.name, indices_rank);

    match &node.inputs[0].ty {
        ArgType::Tensor(input_tensor) => {
            log::debug!(
                "Gather input tensor rank for {}: {}",
                node.name,
                input_tensor.rank
            );
            // Output of rank q+(r-1), where q is rank of indices tensor and r is rank of input
            let output_rank = indices_rank + input_tensor.rank - 1;
            log::debug!("Gather output rank for {}: {}", node.name, output_rank);

            if output_rank == 0 {
                node.outputs[0].ty = ArgType::Scalar(input_tensor.elem_type.clone());
                log::debug!("Gather result for {} is scalar", node.name);
            } else {
                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    elem_type: input_tensor.elem_type.clone(),
                    rank: output_rank,
                    static_shape: None,
                });
                log::debug!(
                    "Gather result for {} is tensor with rank {}",
                    node.name,
                    output_rank
                );
            }
        }
        ArgType::Shape(_) => {
            log::debug!("Gather input is shape for {}", node.name);
            let shape_rank = 1;
            // Output of rank q+(r-1), where q is rank of indices tensor and r is rank of input
            let output_rank = indices_rank + shape_rank - 1;
            log::debug!(
                "Gather output rank for {} with shape input: {}",
                node.name,
                output_rank
            );

            if output_rank == 0 {
                node.outputs[0].ty = ArgType::Scalar(ElementType::Int64);
                log::debug!("Gather result for {} is scalar (from shape)", node.name);
            } else {
                node.outputs[0].ty = ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: output_rank,
                    static_shape: None,
                });
                log::debug!(
                    "Gather result for {} is tensor with rank {} (from shape)",
                    node.name,
                    output_rank
                );
            }
        }
        ty => panic!("Only tensor/shape input is valid but received: {:?}", ty),
    }
}

/// Create a GatherConfig from the attributes of the node
pub fn gather_config(curr: &Node) -> usize {
    // Default: 0 per ONNX spec
    let mut dim: i64 = 0;

    // check if the node has only one input
    if curr.inputs.len() != 2 {
        panic!("Gather: index tensor must be present");
    }

    // extract the shape of the input tensor
    let input_dim = match curr.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor.rank as i64,
        ArgType::Shape(_shape) => 1, // Shape is always 1-D
        other => panic!("Only tensor or shape input is valid, got {:?}", other),
    };

    // extract the attributes
    for (key, value) in curr.attrs.iter() {
        if key.as_str() == "axis" {
            dim = value.clone().into_i64()
        }
    }

    // if dim is negative, it is counted from the end
    if dim < 0 {
        dim += input_dim;
    }

    dim as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axis: i64, input_rank: usize, is_shape: bool) -> Node {
        // Start building the node with the appropriate input type
        let mut builder = NodeBuilder::new(NodeType::Gather, "test_gather").attr_int("axis", axis);

        if is_shape {
            builder = builder.add_input("data", ArgType::Shape(1));
        } else {
            builder = builder.input_tensor_f32("data", input_rank, None);
        }

        // Add indices and output
        builder = builder
            .input_tensor_i64("indices", 1, None)
            .output_tensor_f32("output", input_rank, None);

        builder.build()
    }

    #[test]
    fn test_gather_config_basic() {
        let node = create_test_node(0, 3, false);
        let config = gather_config(&node);
        assert_eq!(config, 0);
    }

    #[test]
    fn test_gather_config_negative_axis() {
        let node = create_test_node(-2, 3, false);
        let config = gather_config(&node);
        assert_eq!(config, 1); // -2 + 3 = 1
    }

    #[test]
    fn test_gather_config_shape_input() {
        let node = create_test_node(0, 0, true);
        let config = gather_config(&node);
        assert_eq!(config, 0);
    }

    #[test]
    #[should_panic(expected = "Gather: index tensor must be present")]
    fn test_gather_config_missing_index() {
        let mut node = create_test_node(0, 3, false);
        node.inputs.pop(); // Remove the indices input
        let _ = gather_config(&node);
    }
}
