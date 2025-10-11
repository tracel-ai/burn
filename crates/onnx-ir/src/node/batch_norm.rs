use crate::processor::{NodeProcessor, ProcessorContext};
use crate::util::same_as_input;

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
    pub fn new(num_features: usize, epsilon: f64, momentum: f64) -> Self {
        Self {
            num_features,
            epsilon,
            momentum,
        }
    }
}

/// Create a BatchNormConfig from the attributes of the node
pub fn batch_norm_config(
    node: &Node,
    graph_data: &mut crate::from_onnx::GraphData,
) -> BatchNormConfig {
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

    BatchNormConfig::new(num_features, epsilon as f64, momentum as f64)
}

pub struct BatchNormProcessor;

impl NodeProcessor for BatchNormProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (6, None)
    }

    fn process(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        _graph_data: &mut crate::from_onnx::GraphData,
    ) {
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
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node(1e-5, 0.9, 64).build_with_graph_data(&mut graph_data);
        let config = batch_norm_config(&node, &mut graph_data);

        assert_eq!(config.num_features, 64);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
        assert!(f64::abs(config.momentum - 0.9) < 1e-6);
    }

    #[test]
    fn test_batch_norm_config_default_values() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node(0.0, 0.0, 32).build_with_graph_data(&mut graph_data);
        let config = batch_norm_config(&node, &mut graph_data);

        assert_eq!(config.num_features, 32);
        assert!(f64::abs(config.epsilon - 0.0) < 1e-6);
        assert!(f64::abs(config.momentum - 0.0) < 1e-6);
    }
}
