use crate::ir::Node;

/// Create a LeakyReluConfig from the alpha attribute of the node
pub fn leaky_relu_config(node: &Node) -> f64 {
    let mut alpha = 0.01;

    for (key, value) in node.attrs.iter() {
        if key.as_str() == "alpha" {
            alpha = value.clone().into_f32() as f64
        }
    }

    alpha
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, AttributeValue, ElementType, NodeType, TensorType};
    use std::collections::HashMap;

    fn create_test_node(alpha: f32) -> Node {
        let inputs = vec![Argument {
            name: "X".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 4,
                static_shape: None,
            }),
            value: None,
            passed: true,
        }];

        let mut attrs = HashMap::new();
        attrs.insert("alpha".to_string(), AttributeValue::Float32(alpha));

        Node {
            node_type: NodeType::LeakyRelu,
            name: "test_leaky_relu".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "Y".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 4,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            }],
            attrs,
        }
    }

    #[test]
    fn test_leaky_relu_config_with_alpha() {
        let node = create_test_node(0.2);
        let alpha = leaky_relu_config(&node);
        assert!((alpha - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu_config_default() {
        let mut node = create_test_node(0.2);
        node.attrs.clear(); // Remove all attributes
        let alpha = leaky_relu_config(&node);
        assert_eq!(alpha, 0.01); // Check default value
    }
}
