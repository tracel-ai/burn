use crate::ir::{ArgType, ElementType, Node, TensorType};

/// Create argmin config from the attributes of the node
pub fn argmin_config(node: &Node) -> usize {
    let mut axis: i64 = 0;

    // check if the node has only one input
    if node.inputs.len() != 1 {
        panic!(
            "ArgMin: multiple inputs are not supported (got {:?})",
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
                        "only select_last_index=0 is supported for argmin in burn. Ignoring supplied value (got {value:?})"
                    );
                }
            }
            "keepdims" => {
                // not all params are supported in burn
                if value.clone().into_i64() != 1 {
                    panic!("Only keepdims=1 is supported for argmin in burn (got {value:?})");
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

/// Update output rank for ArgMin (same as input rank).
pub fn argmin_update_outputs(node: &mut Node) {
    log::debug!("ArgMin rank inference for node {}", node.name);

    if node.inputs.len() != 1 {
        panic!("ArgMin: multiple inputs are not supported");
    }
    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    log::debug!("ArgMin input rank for {}: {}", node.name, tensor.rank);

    // Note: argmin in burn does not support keepdims=false
    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: ElementType::Int64,
        rank: tensor.rank,
        static_shape: None,
    });

    log::debug!("ArgMax output rank for {}: {}", node.name, tensor.rank);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, ElementType, NodeType};
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axis: i64, select_last_index: i64, keepdims: i64) -> Node {
        NodeBuilder::new(NodeType::ArgMax, "test_argmin")
            .input_tensor_f32("data", 3, None)
            .output_tensor_i64("output", 3, None)
            .attr_int("axis", axis)
            .attr_int("select_last_index", select_last_index)
            .attr_int("keepdims", keepdims)
            .build()
    }

    #[test]
    fn test_argmin_config_basic() {
        let node = create_test_node(0, 0, 1);
        let config = argmin_config(&node);
        assert_eq!(config, 0);
    }

    #[test]
    fn test_argmin_config_negative_axis() {
        let node = create_test_node(-2, 0, 1);
        let config = argmin_config(&node);
        assert_eq!(config, 1); // -2 + 3 = 1
    }

    #[test]
    #[should_panic(expected = "ArgMin: multiple inputs are not supported")]
    fn test_argmin_config_multiple_inputs() {
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
        let _ = argmin_config(&node);
    }

    #[test]
    #[should_panic(expected = "Only keepdims=1 is supported for argmin in burn")]
    fn test_argmin_config_keepdims_not_supported() {
        let node = create_test_node(0, 0, 0);
        let _ = argmin_config(&node);
    }
}
