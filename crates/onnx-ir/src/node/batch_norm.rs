use crate::ir::Node;

/// Configuration for BatchNorm operations
#[derive(Debug, Clone)]
pub struct BatchNormConfig {
    /// Number of features (channels)
    pub num_features: usize,
    /// Small constant added for numerical stability
    pub epsilon: f64,
    /// Momentum for running statistics
    pub momentum: f64,
}

impl BatchNormConfig {
    /// Create a new BatchNormConfig
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            epsilon: 1e-5,
            momentum: 0.1,
        }
    }

    /// Set the epsilon value
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set the momentum value
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }
}

/// Create a BatchNormConfig from the attributes of the node
pub fn batch_norm_config(node: &Node) -> BatchNormConfig {
    let weight_shape = node.inputs[1]
        .value
        .as_ref()
        .expect("BatchNorm: weight tensor must be present")
        .shape
        .clone();

    let num_features = weight_shape[0];

    let mut epsilon = 0f32;
    let mut momentum = 0f32;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "momentum" => momentum = value.clone().into_f32(),
            "epsilon" => epsilon = value.clone().into_f32(),
            _ => {}
        }
    }

    BatchNormConfig::new(num_features)
        .with_epsilon(epsilon as f64)
        .with_momentum(momentum as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        ArgType, Argument, AttributeValue, Data, ElementType, NodeType, TensorData, TensorType,
    };
    use std::collections::HashMap;

    fn create_test_node(epsilon: f32, momentum: f32, num_features: usize) -> Node {
        let weight_tensor = TensorData {
            data: Data::Float32s(vec![1.0; num_features]), // Not important for the test
            shape: vec![num_features],
        };

        let bias_tensor = TensorData {
            data: Data::Float32s(vec![0.0; num_features]), // Not important for the test
            shape: vec![num_features],
        };

        let mean_tensor = TensorData {
            data: Data::Float32s(vec![0.0; num_features]), // Not important for the test
            shape: vec![num_features],
        };

        let var_tensor = TensorData {
            data: Data::Float32s(vec![1.0; num_features]), // Not important for the test
            shape: vec![num_features],
        };

        let inputs = vec![
            Argument {
                name: "X".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 4, // NCHW format
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
            Argument {
                name: "mean".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 1,
                    static_shape: None,
                }),
                value: Some(mean_tensor),
                passed: true,
            },
            Argument {
                name: "var".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 1,
                    static_shape: None,
                }),
                value: Some(var_tensor),
                passed: true,
            },
        ];

        let mut attrs = HashMap::new();
        attrs.insert("epsilon".to_string(), AttributeValue::Float32(epsilon));
        attrs.insert("momentum".to_string(), AttributeValue::Float32(momentum));

        Node {
            node_type: NodeType::BatchNormalization,
            name: "test_batchnorm".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "output".to_string(),
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
    fn test_batch_norm_config_basic() {
        let node = create_test_node(1e-5, 0.9, 64);
        let config = batch_norm_config(&node);

        assert_eq!(config.num_features, 64);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
        assert!(f64::abs(config.momentum - 0.9) < 1e-6);
    }

    #[test]
    fn test_batch_norm_config_default_values() {
        let node = create_test_node(0.0, 0.0, 32);
        let config = batch_norm_config(&node);

        assert_eq!(config.num_features, 32);
        assert!(f64::abs(config.epsilon - 0.0) < 1e-6);
        assert!(f64::abs(config.momentum - 0.0) < 1e-6);
    }
}
