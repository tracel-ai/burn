use crate::ir::{ArgType, Node, TensorType};

/// Configuration for Linear operations
#[derive(Debug, Clone)]
pub struct LinearConfig {
    /// Input dimension (features)
    pub d_input: usize,
    /// Output dimension (features)
    pub d_output: usize,
    /// Whether bias is used
    pub bias: bool,
}

impl LinearConfig {
    /// Create a new LinearConfig
    pub fn new(d_input: usize, d_output: usize) -> Self {
        Self {
            d_input,
            d_output,
            bias: true,
        }
    }

    /// Set whether bias is used
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }
}

/// Update output rank for Linear operations (same as input rank).
pub fn linear_update_outputs(node: &mut Node) {
    log::debug!("Linear rank inference for node {}", node.name);

    if let ArgType::Tensor(tensor) = &node.inputs[0].ty {
        log::debug!("Linear input rank for {}: {}", node.name, tensor.rank);

        node.outputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: tensor.elem_type.clone(),
            rank: tensor.rank,
            static_shape: None,
        });

        log::debug!("Linear output rank for {}: {}", node.name, tensor.rank);
    } else {
        panic!("Only tensor input is valid");
    }
}

/// Create a LinearConfig from the attributes of the node
pub fn linear_config(node: &Node) -> LinearConfig {
    if node.inputs.len() < 2 {
        panic!("Linear: missing weight tensor");
    }

    let weight_shape = node.inputs[1]
        .value
        .as_ref()
        .expect("Linear: weight tensor must be present")
        .shape
        .clone();

    // check if the weight tensor has at least 2 dimensions
    if weight_shape.len() < 2 {
        panic!(
            "Linear: weight tensor must have at least 2 dimensions (got {:?})",
            weight_shape.len()
        );
    }

    let (in_size, out_size) = (weight_shape[0], weight_shape[1]);

    // check if the bias is present
    let bias = node.inputs.len() == 3 && node.inputs[2].value.is_some();

    LinearConfig::new(in_size, out_size).with_bias(bias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, Data, ElementType, NodeType, TensorData, TensorType};
    use std::collections::HashMap;

    fn create_test_node(has_bias: bool, weight_dims: Vec<usize>) -> Node {
        let weight_tensor = TensorData {
            data: Data::Float32s(vec![0.0; weight_dims.iter().product()]), // Not important for the test
            shape: weight_dims.clone(),
        };

        let mut inputs = vec![
            Argument {
                name: "input".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 2,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
            Argument {
                name: "weight".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: weight_dims.len(),
                    static_shape: None,
                }),
                value: Some(weight_tensor),
                passed: true,
            },
        ];

        if has_bias {
            let bias_tensor = TensorData {
                data: Data::Float32s(vec![0.0; weight_dims[1]]), // bias size equals output size
                shape: vec![weight_dims[1]],
            };

            inputs.push(Argument {
                name: "bias".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 1,
                    static_shape: None,
                }),
                value: Some(bias_tensor),
                passed: true,
            });
        }

        let attrs = HashMap::new();

        Node {
            node_type: NodeType::Gemm,
            name: "test_linear".to_string(),
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
    fn test_linear_config_basic() {
        let node = create_test_node(false, vec![10, 5]);
        let config = linear_config(&node);

        assert_eq!(config.d_input, 10);
        assert_eq!(config.d_output, 5);
        assert!(!config.bias);
    }

    #[test]
    fn test_linear_config_with_bias() {
        let node = create_test_node(true, vec![10, 5]);
        let config = linear_config(&node);

        assert_eq!(config.d_input, 10);
        assert_eq!(config.d_output, 5);
        assert!(config.bias);
    }

    #[test]
    #[should_panic(expected = "Linear: weight tensor must have at least 2 dimensions")]
    fn test_linear_config_invalid_weight_dims() {
        let node = create_test_node(false, vec![10]);
        let _ = linear_config(&node);
    }

    #[test]
    #[should_panic(expected = "Linear: missing weight tensor")]
    fn test_linear_config_missing_weight() {
        let mut node = create_test_node(false, vec![10, 5]);
        node.inputs.remove(1);
        let _ = linear_config(&node);
    }
}
