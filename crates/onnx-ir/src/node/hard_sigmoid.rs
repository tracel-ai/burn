use crate::ir::{Node, NodeConfig};
use crate::processor::NodeProcessor;
use std::any::Any;

/// Configuration for HardSigmoid operation
#[derive(Debug, Clone)]
pub struct HardSigmoidConfig {
    pub alpha: f64,
    pub beta: f64,
}

impl NodeConfig for HardSigmoidConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct HardSigmoidProcessor;

impl NodeProcessor for HardSigmoidProcessor {
    fn process_config(&self, node: &mut Node, _opset: usize) {
        let mut alpha = 0.2;
        let mut beta = 0.5;

        for (key, value) in node.attrs.iter() {
            match key.as_str() {
                "alpha" => alpha = value.clone().into_f32() as f64,
                "beta" => beta = value.clone().into_f32() as f64,
                _ => {}
            }
        }

        let config = HardSigmoidConfig { alpha, beta };
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

    fn create_test_node(alpha: f32, beta: f32) -> Node {
        NodeBuilder::new(NodeType::HardSigmoid, "test_hard_sigmoid")
            .input_tensor_f32("X", 4, None)
            .output_tensor_f32("Y", 4, None)
            .attr_float("alpha", alpha)
            .attr_float("beta", beta)
            .build()
    }

    #[test]
    fn test_hard_sigmoid_config_with_attrs() {
        let node = create_test_node(0.3, 0.6);
        let mut node = node;
        let processor = HardSigmoidProcessor;
        processor.process_config(&mut node, 16);
        let config = node
            .config
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<HardSigmoidConfig>()
            .unwrap();
        assert!((config.alpha - 0.3).abs() < 1e-6);
        assert!((config.beta - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_hard_sigmoid_config_default() {
        let mut node = create_test_node(0.3, 0.6);
        node.attrs.clear(); // Remove all attributes
        let mut node = node;
        let processor = HardSigmoidProcessor;
        processor.process_config(&mut node, 16);
        let config = node
            .config
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<HardSigmoidConfig>()
            .unwrap();
        assert_eq!(config.alpha, 0.2); // Check default values
        assert_eq!(config.beta, 0.5);
    }
}
