use crate::ir::{Node, NodeConfig};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

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

impl NodeConfig for InstanceNormConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct InstanceNormProcessor;

impl NodeProcessor for InstanceNormProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<Vec<String>, ProcessError> {
        let mut lifted = Vec::new();

        // Lift scale (input 1) and bias (input 2)
        if node.inputs.len() > 1 {
            lifted.push(node.inputs[1].name.clone());
        }
        if node.inputs.len() > 2 {
            lifted.push(node.inputs[2].name.clone());
        }

        Ok(lifted)
    }

    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        const MIN: usize = 6;

        crate::util::validate_opset(opset, MIN)?;
        crate::util::validate_min_inputs(node, 3)?;
        crate::util::validate_output_count(node, 1)?;

        // Validate attributes before extracting config
        for (key, _value) in node.attrs.iter() {
            match key.as_str() {
                "epsilon" => {}
                _ => {
                    return Err(ProcessError::InvalidAttribute {
                        name: key.clone(),
                        reason: format!("Unexpected attribute for InstanceNorm: {key}"),
                    });
                }
            }
        }

        // Output type is same as input
        crate::util::same_as_input(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        let weight_shape = node.inputs[1]
            .into_value()
            .ok_or_else(|| {
                ProcessError::Custom("InstanceNorm: weight tensor must be present".to_string())
            })?
            .shape
            .clone();

        let num_features = weight_shape[0];
        let mut epsilon = 1e-5;

        for (key, value) in node.attrs.iter() {
            if key.as_str() == "epsilon" {
                epsilon = value.clone().into_f32()
            }
        }

        let config = InstanceNormConfig::new(num_features, epsilon as f64);
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(epsilon: f32, num_features: usize) -> NodeBuilder {
        let weight_data = vec![1.0; num_features]; // Not important for the test
        let bias_data = vec![0.0; num_features]; // Not important for the test

        NodeBuilder::new(NodeType::InstanceNormalization, "test_instancenorm")
            .input_tensor_f32("X", 3, None)
            .input_tensor_f32_data("scale", weight_data, vec![num_features])
            .input_tensor_f32_data("bias", bias_data, vec![num_features])
            .output_tensor_f32("output", 3, None)
            .attr_float("epsilon", epsilon)
    }

    #[test]
    fn test_instance_norm_config_basic() {
        let node = create_test_node(1e-5, 64).build_with_graph_data(16);
        let mut node = node;
        let processor = InstanceNormProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<InstanceNormConfig>();

        assert_eq!(config.num_features, 64);
        assert!(f64::abs(config.epsilon - 1e-5) < 1e-6);
    }
}
