use crate::ir::{ArgType, Node};

/// Create concat config from the attributes of the node
pub fn concat_config(node: &Node) -> usize {
    // the axis is the last dimension (Default: 1 per ONNX spec)
    let mut axis: i64 = 1;

    // extract the shape of the input tensor
    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // extract the attributes
    for (key, value) in node.attrs.iter() {
        if key.as_str() == "axis" {
            axis = value.clone().into_i64()
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

    fn create_test_node(axis: i64, input_rank: usize, num_inputs: usize) -> Node {
        let mut inputs = Vec::new();

        // Create multiple inputs for concat
        for i in 0..num_inputs {
            inputs.push(Argument {
                name: format!("data_{}", i),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: input_rank,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            });
        }

        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttributeValue::Int64(axis));

        Node {
            node_type: NodeType::Concat,
            name: "test_concat".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: input_rank,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            }],
            attrs,
        }
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
