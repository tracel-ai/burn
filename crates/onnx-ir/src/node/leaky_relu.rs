use crate::ir::{Node, NodeConfig};
use crate::processor::{NodeProcessor, ProcessorContext};
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

/// Create a LeakyReluConfig from the alpha attribute of the node
pub fn leaky_relu_config(
    node: &Node,
    graph_data: &mut crate::from_onnx::GraphData,
) -> LeakyReluConfig {
    let mut alpha = 0.01;

    for (key, value) in node.attrs.iter() {
        if key.as_str() == "alpha" {
            alpha = value.clone().into_f32() as f64
        }
    }

    LeakyReluConfig { alpha }
}

pub struct LeakyReluProcessor;

impl NodeProcessor for LeakyReluProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (6, None)
    }

    fn process_config(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        graph_data: &mut crate::from_onnx::GraphData,
    ) {
        let config = leaky_relu_config(node, graph_data);
        node.config = Some(Box::new(config));
    }

    fn process_forward(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        _graph_data: &mut crate::from_onnx::GraphData,
    ) {
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
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let config = leaky_relu_config(&node, &mut graph_data);
        assert!((config.alpha - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu_config_default() {
        let mut node = create_test_node(0.2);
        node.attrs.clear(); // Remove all attributes
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let config = leaky_relu_config(&node, &mut graph_data);
        assert_eq!(config.alpha, 0.01); // Check default value
    }
}
