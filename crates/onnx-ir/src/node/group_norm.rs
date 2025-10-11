use crate::processor::{NodeProcessor, ProcessorContext};
use crate::util::same_as_input;

use crate::ir::{Node, NodeConfig};
use std::any::Any;

/// Configuration for GroupNorm operations
#[derive(Debug, Clone)]
pub struct GroupNormConfig {
    /// Number of features (channels)
    pub num_features: usize,
    /// Number of groups
    pub num_groups: usize,
    /// Small constant added for numerical stability
    pub epsilon: f64,
}

impl NodeConfig for GroupNormConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

impl GroupNormConfig {
    /// Create a new GroupNormConfig
    pub fn new(num_features: usize, num_groups: usize, epsilon: f64) -> Self {
        Self {
            num_features,
            num_groups,
            epsilon,
        }
    }
}

pub struct GroupNormProcessor;

impl NodeProcessor for GroupNormProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (18, None)
    }

    fn process_config(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        _graph_data: &mut crate::from_onnx::GraphData,
    ) {
        let weight_shape = node.inputs[1]
            .into_value()
            .as_ref()
            .expect("GroupNorm: weight tensor must be present")
            .shape
            .clone();

        let num_features = weight_shape[0];
        let mut num_groups = None;
        let mut epsilon = 1e-5;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "epsilon" => epsilon = value.clone().into_f32(),
                "num_groups" => num_groups = Some(value.clone().into_i64() as usize),
                "stash_type" => {} // stash_type is read but not used in config
                _ => panic!("Unexpected attribute for GroupNorm: {key}"),
            }
        }

        let num_groups = num_groups.expect("GroupNorm: num_groups attribute must be present");
        if num_groups > 0 && !num_features.is_multiple_of(num_groups) {
            panic!("GroupNorm: number of features must be divisible by the number of groups");
        }

        let config = GroupNormConfig::new(num_features, num_groups, epsilon as f64);
        node.config = Some(Box::new(config));
    }

    fn process_forward(
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

    fn create_test_node(
        epsilon: f32,
        num_features: usize,
        num_groups: usize,
        stash_type: i64,
    ) -> NodeBuilder {
        let weight_data = vec![1.0; num_features]; // Not important for the test
        let bias_data = vec![0.0; num_features]; // Not important for the test

        NodeBuilder::new(NodeType::GroupNormalization, "test_groupnorm")
            .input_tensor_f32("X", 3, None)
            .input_tensor_f32_data("scale", weight_data, vec![num_features])
            .input_tensor_f32_data("bias", bias_data, vec![num_features])
            .output_tensor_f32("output", 3, None)
            .attr_int("num_groups", num_groups as i64)
            .attr_int("stash_type", stash_type)
            .attr_float("epsilon", epsilon)
    }

    #[test]
    fn test_group_norm_config_basic() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let mut node = create_test_node(1e-5, 64, 8, 1).build_with_graph_data(&mut graph_data);
        let processor = GroupNormProcessor;
        let context = ProcessorContext::new(16);
        processor.process_config(&mut node, &context, &mut graph_data);

        let config = node
            .config
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<GroupNormConfig>()
            .unwrap();
        assert_eq!(config.num_features, 64);
        assert_eq!(config.num_groups, 8);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
    }

    #[test]
    fn test_group_norm_config_no_stash_type() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let mut node = create_test_node(1e-5, 64, 8, 0).build_with_graph_data(&mut graph_data);
        let processor = GroupNormProcessor;
        let context = ProcessorContext::new(16);
        processor.process_config(&mut node, &context, &mut graph_data);

        let config = node
            .config
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<GroupNormConfig>()
            .unwrap();
        assert_eq!(config.num_features, 64);
        assert_eq!(config.num_groups, 8);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
    }

    #[test]
    #[should_panic]
    fn test_group_norm_config_invalid_num_groups() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        // num features is not divisible by num groups
        let mut node = create_test_node(1e-5, 64, 7, 0).build_with_graph_data(&mut graph_data);
        let processor = GroupNormProcessor;
        let context = ProcessorContext::new(16);
        processor.process_config(&mut node, &context, &mut graph_data);
    }
}
