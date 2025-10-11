use crate::processor::NodeProcessor;
use crate::util::same_as_input;

use crate::ir::{Node, NodeConfig};
use std::any::Any;

/// Configuration for LayerNorm operations
#[derive(Debug, Clone)]
pub struct LayerNormConfig {
    /// Number of features/model dimension
    pub d_model: usize,
    /// Small constant added for numerical stability
    pub epsilon: f64,
}

impl NodeConfig for LayerNormConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
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

pub struct LayerNormProcessor;

impl NodeProcessor for LayerNormProcessor {
    fn process_config(&self, node: &mut Node, __opset: usize) {
        let weight_shape = node.inputs[1]
            .into_value()
            .expect("LayerNorm: weight tensor must be present")
            .shape
            .clone();

        let num_features = weight_shape[0];

        // When `stash_type` is `1` (default), perform operations in 32-bit float and
        // cast the results back to original dtype
        let mut axis = -1;
        let mut epsilon = 1e-5;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "axis" => axis = value.clone().into_i64(),
                "epsilon" => epsilon = value.clone().into_f32(),
                "stash_type" => {} // stash_type is read but not used in config
                _ => panic!("Unexpected attribute for LayerNorm: {key}"),
            }
        }

        if axis != -1 && axis != weight_shape.len() as i64 - 1 {
            panic!("LayerNorm: normalization is only supported on the last axis right now")
        }

        let config = LayerNormConfig::new(num_features).with_epsilon(epsilon as f64);
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

    fn create_test_node(
        epsilon: f32,
        axis: i64,
        stash_type: i64,
        num_features: usize,
    ) -> NodeBuilder {
        let weight_data = vec![1.0; num_features]; // Not important for the test
        let bias_data = vec![0.0; num_features]; // Not important for the test

        NodeBuilder::new(NodeType::LayerNormalization, "test_layernorm")
            .input_tensor_f32("X", 3, None)
            .input_tensor_f32_data("scale", weight_data, vec![num_features])
            .input_tensor_f32_data("bias", bias_data, vec![num_features])
            .output_tensor_f32("output", 3, None)
            .attr_float("epsilon", epsilon)
            .attr_int("axis", axis)
            .attr_int("stash_type", stash_type)
    }

    #[test]
    fn test_layer_norm_config_basic() {
        let mut node = create_test_node(1e-5, -1, 1, 64).build_with_graph_data(16);
        let processor = LayerNormProcessor;
        processor.process_config(&mut node, 16);

        let config = node
            .config
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<LayerNormConfig>()
            .unwrap();
        assert_eq!(config.d_model, 64);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
    }

    #[test]
    fn test_layer_norm_config_no_stash_type() {
        let mut node = create_test_node(1e-5, -1, 0, 32).build_with_graph_data(16);
        let processor = LayerNormProcessor;
        processor.process_config(&mut node, 16);

        let config = node
            .config
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<LayerNormConfig>()
            .unwrap();
        assert_eq!(config.d_model, 32);
    }

    #[test]
    #[should_panic(
        expected = "LayerNorm: normalization is only supported on the last axis right now"
    )]
    fn test_layer_norm_config_invalid_axis() {
        // For a 1D weight tensor with shape [num_features],
        // both axis=0 (the first and only dim) and axis=-1 (the last dim) are valid
        // So we need to use a 2D weight tensor to test the invalid axis case

        // Create a custom node with a 2D weight tensor
        let weight_data = vec![1.0; 32 * 64]; // 2D weight tensor
        let bias_data = vec![0.0; 32 * 64];

        let node = NodeBuilder::new(NodeType::LayerNormalization, "test_layernorm_invalid")
            .input_tensor_f32("X", 3, None)
            .input_tensor_f32_data("scale", weight_data, vec![32, 64]) // 2D shape
            .input_tensor_f32_data("bias", bias_data, vec![32, 64])
            .output_tensor_f32("output", 3, None)
            .attr_float("epsilon", 1e-5)
            .attr_int("axis", 0) // axis=0 is NOT the last dimension for 2D weight
            .attr_int("stash_type", 1)
            .build_with_graph_data(16);

        // Now axis=0 should trigger a panic since it's not the last dimension (1)
        let mut node = node;
        let processor = LayerNormProcessor;
        processor.process_config(&mut node, 16);
    }
}
