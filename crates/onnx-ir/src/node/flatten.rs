use crate::ir::{ArgType, Node, TensorType};

/// Update output type for Flatten operation (rank 2).
pub fn flatten_update_outputs(node: &mut Node) {
    if node.inputs.len() != 1 {
        panic!("Flatten: multiple inputs are not supported");
    }
    let tensor = node
        .inputs
        .iter()
        .find_map(|input| match &input.ty {
            ArgType::Tensor(tensor) => Some(tensor),
            _ => None,
        })
        .unwrap();

    // Flatten to a 2D tensor
    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: 2,
        ..tensor.clone()
    });
}

/// Create a FlattenConfig from the attributes of the node
pub fn flatten_config(curr: &Node) -> usize {
    // the begin dimension is the first dimension (Default: 1 per ONNX spec)
    let mut axis: i64 = 1;

    // check if the node has only one input
    if curr.inputs.len() != 1 {
        panic!(
            "Flatten: multiple inputs are not supported (got {:?})",
            curr.inputs.len()
        );
    }

    // extract the shape of the input tensor
    let tensor = match curr.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // check if the input tensor has at least 2 dimensions
    if tensor.rank < 2 {
        panic!(
            "Flatten: input tensor must have at least 2 dimensions (got {:?})",
            tensor.rank
        );
    }

    // extract the attributes
    for (key, value) in curr.attrs.iter() {
        if key.as_str() == "axis" {
            axis = value.clone().into_i64()
        }
    }

    // if beg_dim is negative, it is counted from the end
    if axis < 0 {
        axis += tensor.rank as i64;
    }

    axis as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, AttributeValue, ElementType, NodeType, TensorType};
    use std::collections::HashMap;

    fn create_test_node(axis: i64) -> Node {
        let inputs = vec![Argument {
            name: "data".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 4,
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttributeValue::Int64(axis));

        Node {
            node_type: NodeType::Flatten,
            name: "test_flatten".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 2,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            }],
            attrs,
        }
    }

    #[test]
    fn test_flatten_config_basic() {
        let node = create_test_node(1);
        let config = flatten_config(&node);
        assert_eq!(config, 1);
    }

    #[test]
    fn test_flatten_config_with_negative_axis() {
        let node = create_test_node(-2);
        let config = flatten_config(&node);
        assert_eq!(config, 2); // -2 + 4 = 2
    }

    #[test]
    #[should_panic(expected = "Flatten: input tensor must have at least 2 dimensions")]
    fn test_flatten_config_with_low_rank() {
        let mut node = create_test_node(1);
        node.inputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: ElementType::Float32,
            rank: 1,
            static_shape: None,
        });
        let _ = flatten_config(&node);
    }

    #[test]
    #[should_panic(expected = "Flatten: multiple inputs are not supported")]
    fn test_flatten_config_with_multiple_inputs() {
        let mut node = create_test_node(1);
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
        let _ = flatten_config(&node);
    }
}
