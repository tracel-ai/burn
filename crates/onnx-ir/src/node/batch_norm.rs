use crate::ir::{Node, NodeConfig};
use crate::processor::NodeProcessor;
use crate::util::validate_opset;

use crate::util::same_as_input;
use std::any::Any;

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
    pub fn new(num_features: usize, epsilon: f64, momentum: f64) -> Self {
        Self {
            num_features,
            epsilon,
            momentum,
        }
    }
}

impl NodeConfig for BatchNormConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct BatchNormProcessor;

impl NodeProcessor for BatchNormProcessor {
    fn process_config(&self, node: &mut Node, opset: usize) {
        // BatchNormalization implementation supports opset 9+ (for multiple outputs)
        validate_opset(&node.node_type, opset, 9);

        let weight_shape = node.inputs[1]
            .into_value()
            .expect("BatchNorm: weight tensor must be present")
            .shape;

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

        let config = BatchNormConfig::new(num_features, epsilon as f64, momentum as f64);
        node.config = Some(Box::new(config));
    }

    fn first_pass(&self, node: &mut Node, _opset: usize) {
        same_as_input(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(epsilon: f32, momentum: f32, num_features: usize) -> NodeBuilder {
        let ones = vec![1.0; num_features];
        let zeros = vec![0.0; num_features];

        NodeBuilder::new(NodeType::BatchNormalization, "test_batchnorm")
            .input_tensor_f32("X", 4, None) // NCHW format
            .input_tensor_f32_data("scale", ones.clone(), vec![num_features])
            .input_tensor_f32_data("bias", zeros.clone(), vec![num_features])
            .input_tensor_f32_data("mean", zeros.clone(), vec![num_features])
            .input_tensor_f32_data("var", ones.clone(), vec![num_features])
            .output_tensor_f32("output", 4, None)
            .attr_float("epsilon", epsilon)
            .attr_float("momentum", momentum)
    }

    #[test]
    fn test_batch_norm_config_basic() {
        let node = create_test_node(1e-5, 0.9, 64).build_with_graph_data(16);
        let mut node = node;
        let processor = BatchNormProcessor;
        processor.process_config(&mut node, 16);
        let config = node.config::<BatchNormConfig>();

        assert_eq!(config.num_features, 64);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
        assert!(f64::abs(config.momentum - 0.9) < 1e-6);
    }

    #[test]
    fn test_batch_norm_config_default_values() {
        let node = create_test_node(0.0, 0.0, 32).build_with_graph_data(16);
        let mut node = node;
        let processor = BatchNormProcessor;
        processor.process_config(&mut node, 16);
        let config = node.config::<BatchNormConfig>();

        assert_eq!(config.num_features, 32);
        assert!(f64::abs(config.epsilon - 0.0) < 1e-6);
        assert!(f64::abs(config.momentum - 0.0) < 1e-6);
    }
}
