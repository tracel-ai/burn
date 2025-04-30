use crate::ir::Node;

/// Create a HardSigmoidConfig from the alpha and beta attributes of the node
pub fn hard_sigmoid_config(node: &Node) -> (f64, f64) {
    let mut alpha = 0.2;
    let mut beta = 0.5;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "alpha" => alpha = value.clone().into_f32() as f64,
            "beta" => beta = value.clone().into_f32() as f64,
            _ => {}
        }
    }

    (alpha, beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, AttributeValue, ElementType, NodeType, TensorType};
    use std::collections::HashMap;

    fn create_test_node(alpha: f32, beta: f32) -> Node {
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
        attrs.insert("beta".to_string(), AttributeValue::Float32(beta));

        Node {
            node_type: NodeType::HardSigmoid,
            name: "test_hard_sigmoid".to_string(),
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
    fn test_hard_sigmoid_config_with_attrs() {
        let node = create_test_node(0.3, 0.6);
        let (alpha, beta) = hard_sigmoid_config(&node);
        assert!((alpha - 0.3).abs() < 1e-6);
        assert!((beta - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_hard_sigmoid_config_default() {
        let mut node = create_test_node(0.3, 0.6);
        node.attrs.clear(); // Remove all attributes
        let (alpha, beta) = hard_sigmoid_config(&node);
        assert_eq!(alpha, 0.2); // Check default values
        assert_eq!(beta, 0.5);
    }
}
