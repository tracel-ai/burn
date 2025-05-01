use crate::ir::Node;

/// Configuration for LayerNorm operations
#[derive(Debug, Clone)]
pub struct LayerNormConfig {
    /// Number of features/model dimension
    pub d_model: usize,
    /// Small constant added for numerical stability
    pub epsilon: f64,
}

impl LayerNormConfig {
    /// Create a new LayerNormConfig
    pub fn new(d_model: usize) -> Self {
        Self {
            d_model,
            epsilon: 1e-5,
        }
    }

    /// Set the epsilon value
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }
}

/// Create a LayerNormConfig from the attributes of the node
pub fn layer_norm_config(node: &Node) -> (LayerNormConfig, bool) {
    let weight_shape = node.inputs[1]
        .value
        .as_ref()
        .expect("LayerNorm: weight tensor must be present")
        .shape
        .clone();

    let num_features = weight_shape[0];

    // When `stash_type` is `1` (default), perform operations in 32-bit float and
    // cast the results back to original dtype
    let mut stash_type = 1;
    let mut axis = -1;
    let mut epsilon = 1e-5;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "axis" => axis = value.clone().into_i64(),
            "epsilon" => epsilon = value.clone().into_f32(),
            "stash_type" => stash_type = value.clone().into_i64(),
            _ => {}
        }
    }

    if axis != -1 && axis != weight_shape.len() as i64 - 1 {
        panic!("LayerNorm: normalization is only supported on the last axis right now")
    }

    (
        LayerNormConfig::new(num_features).with_epsilon(epsilon as f64),
        stash_type == 1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        ArgType, Argument, AttributeValue, Data, ElementType, NodeType, TensorData, TensorType,
    };
    use std::collections::HashMap;

    fn create_test_node(epsilon: f32, axis: i64, stash_type: i64, num_features: usize) -> Node {
        let weight_tensor = TensorData {
            data: Data::Float32s(vec![1.0; num_features]), // Not important for the test
            shape: vec![num_features],
        };

        let bias_tensor = TensorData {
            data: Data::Float32s(vec![0.0; num_features]), // Not important for the test
            shape: vec![num_features],
        };

        let inputs = vec![
            Argument {
                name: "X".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 3,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
            Argument {
                name: "scale".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 1,
                    static_shape: None,
                }),
                value: Some(weight_tensor),
                passed: true,
            },
            Argument {
                name: "bias".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 1,
                    static_shape: None,
                }),
                value: Some(bias_tensor),
                passed: true,
            },
        ];

        let mut attrs = HashMap::new();
        attrs.insert("epsilon".to_string(), AttributeValue::Float32(epsilon));
        attrs.insert("axis".to_string(), AttributeValue::Int64(axis));
        attrs.insert("stash_type".to_string(), AttributeValue::Int64(stash_type));

        Node {
            node_type: NodeType::LayerNormalization,
            name: "test_layernorm".to_string(),
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
    fn test_layer_norm_config_basic() {
        let node = create_test_node(1e-5, -1, 1, 64);
        let (config, stash_type_flag) = layer_norm_config(&node);

        assert_eq!(config.d_model, 64);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
        assert!(stash_type_flag);
    }

    #[test]
    fn test_layer_norm_config_no_stash_type() {
        let node = create_test_node(1e-5, -1, 0, 32);
        let (config, stash_type_flag) = layer_norm_config(&node);

        assert_eq!(config.d_model, 32);
        assert!(!stash_type_flag);
    }

    #[test]
    #[should_panic]
    fn test_layer_norm_config_invalid_axis() {
        // For a 1D weight tensor with shape [num_features],
        // both axis=0 (the first and only dim) and axis=-1 (the last dim) are valid
        // So we need to use a 2D weight tensor to test the invalid axis case

        // Create a custom node with a 2D weight tensor
        let mut node = create_test_node(1e-5, 0, 1, 64);

        // Modify the weight tensor to be 2D
        if let Some(ref mut tensor) = node.inputs[1].value {
            tensor.shape = vec![64, 64]; // Make it 2D
        }

        // Now axis=0 should trigger a panic since it's not the last dimension
        let _ = layer_norm_config(&node);
    }
}
