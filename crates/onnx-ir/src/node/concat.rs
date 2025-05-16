use crate::ir::{ArgType, Node, TensorType};

/// Update output rank for Concat (same as first tensor input).
pub fn concat_update_outputs(node: &mut Node) {
    log::debug!("Concat rank inference for node {}", node.name);

    let tensor = node
        .inputs
        .iter()
        .find_map(|input| match &input.ty {
            ArgType::Tensor(tensor) => Some(tensor.clone()),
            _ => None,
        })
        .unwrap();

    log::debug!("Concat using input rank for {}: {}", node.name, tensor.rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type,
        rank: tensor.rank,
        static_shape: None,
    });

    log::debug!("Concat output rank for {}: {}", node.name, tensor.rank);
}

/// Create concat config from the attributes of the node
#[must_use]
pub fn concat_config(node: &Node) -> usize {
    // the axis is the last dimension (Default: 1 per ONNX spec)
    let mut axis: i64 = 1;

    // extract the shape of the input tensor
    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // extract the attributes
    for (key, value) in &node.attrs {
        if key.as_str() == "axis" {
            axis = value.clone().into_i64();
        }
    }

    // if axis is negative, it is counted from the end
    if axis < 0 {
        axis += tensor.rank as i64;
    }

    axis as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axis: i64, input_rank: usize, num_inputs: usize) -> Node {
        NodeBuilder::new(NodeType::Concat, "test_concat")
            .input_tensors_f32::<Vec<usize>>("data", num_inputs, input_rank, None)
            .output_tensor_f32("output", input_rank, None)
            .attr_int("axis", axis)
            .build()
    }

    #[test]
    fn test_concat_config_basic() {
        let node = create_test_node(1, 3, 2);
        let config = concat_config(&node);
        assert_eq!(config, 1);
    }

    #[test]
    fn test_concat_config_negative_axis() {
        let node = create_test_node(-2, 3, 2);
        let config = concat_config(&node);
        assert_eq!(config, 1); // -2 + 3 = 1
    }

    #[test]
    #[should_panic(expected = "Only tensor input is valid")]
    fn test_concat_config_invalid_input() {
        let mut node = create_test_node(1, 3, 1);
        node.inputs[0].ty = ArgType::Shape(1);
        let _ = concat_config(&node);
    }
}
