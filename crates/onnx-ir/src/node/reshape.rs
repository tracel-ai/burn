use crate::ir::{Node, TensorData};

pub fn reshape_config(node: &Node) -> Vec<i64> {
    let mut allowzero = 0;

    for (key, value) in node.attrs.iter() {
        if key.as_str() == "allowzero" {
            allowzero = value.clone().into_i64()
        }
    }

    // Burn does not support zero size shape (0 means false in ONNX)
    // (see https://onnx.ai/onnx/operators/onnx__Reshape.html#attributes)
    if allowzero != 0 {
        panic!("Zero shape size is not supported");
    }

    // TODO: check "shape" attribute
    if node.inputs.len() != 2 || node.inputs[1].value.is_none() {
        panic!("Reshape: shape tensor must be present for {:?}", node);
    }

    match &node.inputs[1].value {
        Some(TensorData { data, shape, .. }) => {
            assert_eq!(shape.len(), 1, "Reshape: shape tensor must be 1D");
            data.clone().into_i64s()
        }
        _ => panic!("Only tensor input is valid for shape"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        ArgType, Argument, AttributeValue, Data, ElementType, NodeType, TensorData, TensorType,
    };
    use std::collections::HashMap;

    fn create_test_node(allowzero: i64, shape_vec: Vec<i64>) -> Node {
        let inputs = vec![
            Argument {
                name: "data".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 4,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
            Argument {
                name: "shape".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Int64,
                    rank: 1,
                    static_shape: None,
                }),
                value: Some(TensorData {
                    data: Data::Int64s(shape_vec),
                    shape: vec![2],
                }),
                passed: true,
            },
        ];

        let mut attrs = HashMap::new();
        if allowzero != 0 {
            attrs.insert("allowzero".to_string(), AttributeValue::Int64(allowzero));
        }

        Node {
            node_type: NodeType::Reshape,
            name: "test_reshape".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "reshaped".to_string(),
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
    fn test_reshape_config_basic() {
        let node = create_test_node(0, vec![2, 3]);
        let shape = reshape_config(&node);
        assert_eq!(shape, vec![2, 3]);
    }

    #[test]
    #[should_panic(expected = "Zero shape size is not supported")]
    fn test_reshape_config_allowzero_not_supported() {
        let node = create_test_node(1, vec![2, 3]);
        let _ = reshape_config(&node);
    }

    #[test]
    #[should_panic(expected = "shape tensor must be present")]
    fn test_reshape_config_no_shape_input() {
        let mut node = create_test_node(0, vec![2, 3]);
        node.inputs.pop(); // Remove the shape input
        let _ = reshape_config(&node);
    }

    #[test]
    #[should_panic(expected = "shape tensor must be 1D")]
    fn test_reshape_config_invalid_shape_dim() {
        let mut node = create_test_node(0, vec![2, 3]);
        // Modify the shape tensor's shape to be 2D
        if let Some(tensor_data) = &mut node.inputs[1].value {
            tensor_data.shape = vec![2, 1];
        }
        let _ = reshape_config(&node);
    }
}
