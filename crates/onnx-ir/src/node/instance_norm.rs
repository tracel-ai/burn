use crate::ir::Node;

/// Configuration for InstanceNorm operations
#[derive(Debug, Clone)]
pub struct InstanceNormConfig {
    /// Number of features (channels)
    pub num_features: usize,
    /// Small constant added for numerical stability
    pub epsilon: f64,
}

impl InstanceNormConfig {
    /// Create a new InstanceNormConfig
    pub fn new(num_features: usize, epsilon: f64) -> Self {
        Self {
            num_features,
            epsilon,
        }
    }
}

/// Create a InstanceNormConfig from the attributes of the node
pub fn instance_norm_config(node: &Node) -> InstanceNormConfig {
    let weight_shape = node.inputs[1]
        .value
        .as_ref()
        .expect("InstanceNorm: weight tensor must be present")
        .shape
        .clone();

    let num_features = weight_shape[0];
    let mut epsilon = 1e-5;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "epsilon" => epsilon = value.clone().into_f32(),
            _ => panic!("Unexpected attribute for InstanceNorm: {key}"),
        }
    }

    InstanceNormConfig::new(num_features, epsilon as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(epsilon: f32, num_features: usize) -> Node {
        let weight_data = vec![1.0; num_features]; // Not important for the test
        let bias_data = vec![0.0; num_features]; // Not important for the test

        NodeBuilder::new(NodeType::InstanceNormalization, "test_instancenorm")
            .input_tensor_f32("X", 3, None)
            .input_tensor_f32_data("scale", weight_data, vec![num_features])
            .input_tensor_f32_data("bias", bias_data, vec![num_features])
            .output_tensor_f32("output", 3, None)
            .attr_float("epsilon", epsilon)
            .build()
    }

    #[test]
    fn test_instance_norm_config_basic() {
        let node = create_test_node(1e-5, 64);
        let config = instance_norm_config(&node);

        assert_eq!(config.num_features, 64);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
    }
}
