use crate::ir::{ArgType, Node};

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
                        "only select_last_index=0 is supported for argmax in burn. Ignoring supplied value (got {:?})",
                        value
                    );
                }
            }
            "keepdims" => {
                // not all params are supported in burn
                if value.clone().into_i64() != 1 {
                    panic!(
                        "Only keepdims=1 is supported for argmax in burn (got {:?})",
                        value
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, AttributeValue, ElementType, NodeType, TensorType};
    use std::collections::HashMap;

    fn create_test_node(axis: i64, select_last_index: i64, keepdims: i64) -> Node {
        let inputs = vec![Argument {
            name: "data".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 3,
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttributeValue::Int64(axis));
        attrs.insert(
            "select_last_index".to_string(),
            AttributeValue::Int64(select_last_index),
        );
        attrs.insert("keepdims".to_string(), AttributeValue::Int64(keepdims));

        Node {
            node_type: NodeType::ArgMax,
            name: "test_argmax".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: 3,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            }],
            attrs,
        }
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
    #[should_panic(expected = "Only keepdims=1 is supported for argmax in burn")]
    fn test_argmax_config_keepdims_not_supported() {
        let node = create_test_node(0, 0, 0);
        let _ = argmax_config(&node);
    }
}
