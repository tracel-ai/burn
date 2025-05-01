use crate::ir::{Data, Node};

/// Configuration for Dropout operations
#[derive(Debug, Clone)]
pub struct DropoutConfig {
    /// Probability of dropping out a unit
    pub prob: f64,
}

impl DropoutConfig {
    /// Create a new DropoutConfig
    pub fn new(prob: f64) -> Self {
        Self { prob }
    }
}

/// Create a DropoutConfig from an attribute and state of the node
pub fn dropout_config(node: &Node) -> DropoutConfig {
    // Opset 7 and older store probability as an attribute
    if node.attrs.contains_key("ratio") {
        let prob = node.attrs.get("ratio").unwrap().clone().into_f32();
        return DropoutConfig::new(prob as f64);
    }

    if node.inputs.len() < 2 {
        panic!("Dropout configuration must have at least 2 inputs");
    }

    let ratio = node.inputs[1]
        .value
        .clone()
        .expect("Dropout ratio must be passed in the second input")
        .data
        .into_scalar();

    let prob = match ratio {
        Data::Float16(ratio) => f64::from(f32::from(ratio)),
        Data::Float32(ratio) => ratio as f64,
        Data::Float64(ratio) => ratio,
        _ => panic!("Dropout ratio must be a float"),
    };

    DropoutConfig::new(prob)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        ArgType, Argument, AttributeValue, Data, ElementType, NodeType, TensorData, TensorType,
    };
    use std::collections::HashMap;

    fn create_test_node_with_attr(ratio: f32) -> Node {
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
        attrs.insert("ratio".to_string(), AttributeValue::Float32(ratio));

        Node {
            node_type: NodeType::Dropout,
            name: "test_dropout".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 3,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            }],
            attrs,
        }
    }

    fn create_test_node_with_input(ratio: f32) -> Node {
        let ratio_tensor = TensorData {
            data: Data::Float32(ratio),
            shape: vec![],
        };

        let inputs = vec![
            Argument {
                name: "data".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 3,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
            Argument {
                name: "ratio".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 0,
                    static_shape: None,
                }),
                value: Some(ratio_tensor),
                passed: true,
            },
        ];

        let attrs = HashMap::new();

        Node {
            node_type: NodeType::Dropout,
            name: "test_dropout".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
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
    fn test_dropout_config_with_attr() {
        let node = create_test_node_with_attr(0.3);
        let config = dropout_config(&node);
        assert!(f64::abs(config.prob - 0.3) < 1e-6);
    }

    #[test]
    fn test_dropout_config_with_input() {
        let node = create_test_node_with_input(0.5);
        let config = dropout_config(&node);
        assert!(f64::abs(config.prob - 0.5) < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Dropout configuration must have at least 2 inputs")]
    fn test_dropout_config_missing_input() {
        let mut node = create_test_node_with_input(0.5);
        node.attrs = HashMap::new(); // Remove attributes
        node.inputs.remove(1); // Remove ratio input
        let _ = dropout_config(&node);
    }
}
