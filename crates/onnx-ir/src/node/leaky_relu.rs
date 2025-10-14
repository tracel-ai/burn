use crate::ir::{Node, NodeConfig};
use crate::processor::NodeProcessor;
use crate::util::validate_opset;

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
    fn process_config(&self, node: &mut Node, opset: usize) {
        // LeakyRelu implementation supports opset 6+ (for shape inference)
        validate_opset(&node.node_type, opset, 6);

        // ALL logic from leaky_relu_config inlined here
        let mut alpha = 0.01;

        for (key, value) in node.attrs.iter() {
            if key.as_str() == "alpha" {
                alpha = value.clone().into_f32() as f64
            }
        }

        let config = LeakyReluConfig { alpha };
        node.config = Some(Box::new(config));
    }

    fn first_pass(&self, node: &mut Node, _opset: usize) {
        crate::util::same_as_input(node);
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
        processor.process_config(&mut node, 16);
        let config = node.config::<LeakyReluConfig>();
        assert!((config.alpha - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu_config_default() {
        let mut node = create_test_node(0.2);
        node.attrs.clear(); // Remove all attributes
        let mut node = node;
        let processor = LeakyReluProcessor;
        processor.process_config(&mut node, 16);
        let config = node.config::<LeakyReluConfig>();
        assert_eq!(config.alpha, 0.01); // Check default value
    }
}
