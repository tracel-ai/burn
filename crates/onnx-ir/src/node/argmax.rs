use crate::ir::{ArgType, ElementType, Node, TensorType};

/// Create argmax config from the attributes of the node
pub fn argmax_config(node: &Node) -> usize {
    let mut axis: i64 = 0;

    // check if the node has only one input
    if node.inputs.len() != 1 {
        panic!(
            "Argmax: multiple inputs are not supported (got {:?})",
            node.inputs.len()
        );
    }

    // extract the shape of the input tensor
    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // extract the attributes
    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axis" => axis = value.clone().into_i64(),
            "select_last_index" => {
                // not all params are supported in burn
                if value.clone().into_i64() != 0 {
                    log::warn!(
                        "only select_last_index=0 is supported for argmax in burn. Ignoring supplied value (got {value:?})"
                    );
                }
            }
            "keepdims" => {
                // keepdims=0 and keepdims=1 are both supported
                let keepdims_val = value.clone().into_i64();
                if keepdims_val != 0 && keepdims_val != 1 {
                    panic!(
                        "Only keepdims=0 or keepdims=1 is supported for argmax in burn (got {value:?})",
                    );
                }
            }
            _ => {}
        }
    }

    // if axis is negative, it is counted from the end
    if axis < 0 {
        axis += tensor.rank as i64;
    }

    axis as usize
}

/// Update output rank for ArgMax based on keepdims parameter.
pub fn argmax_update_outputs(node: &mut Node) {
    log::debug!("ArgMax rank inference for node {}", node.name);

    if node.inputs.len() != 1 {
        panic!("ArgMax: multiple inputs are not supported");
    }
    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    log::debug!("ArgMax input rank for {}: {}", node.name, tensor.rank);

    // Check keepdims attribute to determine output rank
    let mut keepdims = 1; // default value per ONNX spec
    for (key, value) in &node.attrs {
        if key == "keepdims" {
            keepdims = value.clone().into_i64();
            break;
        }
    }

    // For burn compatibility, argmax always outputs a tensor
    // When keepdims=0, we still output a tensor but with adjusted rank
    let output_rank = if keepdims == 1 {
        // keepdims=1: output rank same as input rank (dimension becomes 1)
        tensor.rank
    } else {
        // keepdims=0: output rank is input rank - 1 (dimension is removed)
        // But ensure minimum rank of 1 for burn compatibility
        if tensor.rank == 0 {
            panic!("Cannot reduce rank 0 tensor with keepdims=0");
        }
        (tensor.rank - 1).max(1)
    };

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: ElementType::Int64,
        rank: output_rank,
        static_shape: None,
    });

    log::debug!(
        "ArgMax output rank for {} (keepdims={}): {}",
        node.name,
        keepdims,
        output_rank
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, ElementType, NodeType};
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axis: i64, select_last_index: i64, keepdims: i64) -> Node {
        NodeBuilder::new(NodeType::ArgMax, "test_argmax")
            .input_tensor_f32("data", 3, None)
            .output_tensor_i64("output", 3, None)
            .attr_int("axis", axis)
            .attr_int("select_last_index", select_last_index)
            .attr_int("keepdims", keepdims)
            .build()
    }

    #[test]
    fn test_argmax_config_basic() {
        let node = create_test_node(0, 0, 1);
        let config = argmax_config(&node);
        assert_eq!(config, 0);
    }

    #[test]
    fn test_argmax_config_negative_axis() {
        let node = create_test_node(-2, 0, 1);
        let config = argmax_config(&node);
        assert_eq!(config, 1); // -2 + 3 = 1
    }

    #[test]
    #[should_panic(expected = "Argmax: multiple inputs are not supported")]
    fn test_argmax_config_multiple_inputs() {
        let mut node = create_test_node(0, 0, 1);
        node.inputs.push(Argument {
            name: "extra".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 1,
                static_shape: None,
            }),
            value: None,
            passed: true,
        });
        let _ = argmax_config(&node);
    }

    #[test]
    fn test_argmax_config_keepdims_supported() {
        let node_keepdims_0 = create_test_node(0, 0, 0);
        let config_0 = argmax_config(&node_keepdims_0);
        assert_eq!(config_0, 0);

        let node_keepdims_1 = create_test_node(0, 0, 1);
        let config_1 = argmax_config(&node_keepdims_1);
        assert_eq!(config_1, 0);
    }

    #[test]
    #[should_panic(expected = "Only keepdims=0 or keepdims=1 is supported for argmax in burn")]
    fn test_argmax_config_keepdims_invalid() {
        let node = create_test_node(0, 0, 2); // Invalid keepdims value
        let _ = argmax_config(&node);
    }

    #[test]
    fn test_argmax_update_outputs_keepdims_0() {
        // Test argmax with keepdims=0 - output rank should be reduced but minimum 1 for burn
        let mut node = NodeBuilder::new(NodeType::ArgMax, "test_argmax_keepdims_0")
            .attr_int("axis", 1)
            .attr_int("keepdims", 0)
            .input_tensor_f32("data", 2, None) // 2D input
            .output_tensor_i64("output", 2, None) // Will be updated by argmax_update_outputs
            .build();

        argmax_update_outputs(&mut node);

        // Should output tensor with rank 1 (2 - 1 = 1, max(1, 1) = 1)
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 1);
                assert_eq!(tensor.elem_type, crate::ir::ElementType::Int64);
            }
            other => panic!("Expected tensor output, got {:?}", other),
        }
    }

    #[test]
    fn test_argmax_update_outputs_keepdims_1() {
        // Test argmax with keepdims=1 - output rank should be same as input
        let mut node = NodeBuilder::new(NodeType::ArgMax, "test_argmax_keepdims_1")
            .attr_int("axis", 0)
            .attr_int("keepdims", 1)
            .input_tensor_f32("data", 3, None) // 3D input
            .output_tensor_i64("output", 3, None) // Will be updated by argmax_update_outputs
            .build();

        argmax_update_outputs(&mut node);

        // Should output tensor with same rank as input (3)
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 3);
                assert_eq!(tensor.elem_type, crate::ir::ElementType::Int64);
            }
            other => panic!("Expected tensor output, got {:?}", other),
        }
    }

    #[test]
    fn test_argmax_update_outputs_keepdims_0_min_rank() {
        // Test argmax with keepdims=0 on 1D tensor - should maintain rank 1 for burn compatibility
        let mut node = NodeBuilder::new(NodeType::ArgMax, "test_argmax_1d_keepdims_0")
            .attr_int("axis", 0)
            .attr_int("keepdims", 0)
            .input_tensor_f32("data", 1, None) // 1D input
            .output_tensor_i64("output", 1, None) // Will be updated by argmax_update_outputs
            .build();

        argmax_update_outputs(&mut node);

        // Should output tensor with rank 1 (1 - 1 = 0, max(0, 1) = 1)
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 1);
                assert_eq!(tensor.elem_type, crate::ir::ElementType::Int64);
            }
            other => panic!("Expected tensor output, got {:?}", other),
        }
    }
}
