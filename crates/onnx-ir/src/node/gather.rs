use crate::ir::{ArgType, Node};

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
    use crate::ir::{Argument, AttributeValue, ElementType, NodeType, TensorType};
    use std::collections::HashMap;

    fn create_test_node(axis: i64, input_rank: usize, is_shape: bool) -> Node {
        let input_ty = if is_shape {
            ArgType::Shape(1)
        } else {
            ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: input_rank,
                static_shape: None,
            })
        };

        let inputs = vec![
            Argument {
                name: "data".to_string(),
                ty: input_ty,
                value: None,
                passed: true,
            },
            Argument {
                name: "indices".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: 1,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
        ];

        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), AttributeValue::Int64(axis));

        Node {
            node_type: NodeType::Gather,
            name: "test_gather".to_string(),
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
