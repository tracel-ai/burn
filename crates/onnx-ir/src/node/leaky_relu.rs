use crate::ir::{Node, NodeConfig};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

use std::any::Any;

/// Configuration for LeakyRelu operations
#[derive(Debug, Clone)]
pub struct LeakyReluConfig {
    /// Alpha value for negative slope
    pub alpha: f64,
}

impl NodeConfig for LeakyReluConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct LeakyReluProcessor;

impl NodeProcessor for LeakyReluProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 6)?;
        crate::processor::validate_input_count(node, 1)?;
        crate::processor::validate_output_count(node, 1)?;

        // Output type is same as input
        crate::processor::same_as_input(node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Extract alpha attribute
        let mut alpha = 0.01;
        for (key, value) in node.attrs.iter() {
            if key.as_str() == "alpha" {
                alpha = value.clone().into_f32() as f64
            }
        }

        let config = LeakyReluConfig { alpha };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(alpha: f32) -> Node {
        NodeBuilder::new(NodeType::LeakyRelu, "test_leaky_relu")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_float("alpha", alpha)
            .build()
    }

    #[test]
    fn test_leaky_relu_config_with_alpha() {
        let node = create_test_node(0.2);
        let mut node = node;
        let processor = LeakyReluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<LeakyReluConfig>();
        assert!((config.alpha - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu_config_default() {
        let mut node = create_test_node(0.2);
        node.attrs.clear(); // Remove all attributes
        let mut node = node;
        let processor = LeakyReluProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
        let config = node.config::<LeakyReluConfig>();
        assert_eq!(config.alpha, 0.01); // Check default value
    }
}
