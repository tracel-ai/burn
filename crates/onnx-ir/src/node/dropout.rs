use crate::processor::{NodeProcessor, ProcessorContext};
use crate::util::same_as_input;

use crate::ir::{Data, Node, NodeConfig};
use std::any::Any;

/// Configuration for Dropout operations
#[derive(Debug, Clone)]
pub struct DropoutConfig {
    /// Probability of dropping out a unit
    pub prob: f64,
}

impl DropoutConfig {
    /// Create a new DropoutConfig
    pub fn new(prob: f64) -> Self {
        Self { prob }
    }
}

impl NodeConfig for DropoutConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

/// Create a DropoutConfig from an attribute and state of the node
pub fn dropout_config(node: &Node, graph_data: &mut crate::from_onnx::GraphData) -> DropoutConfig {
    // Opset 7 and older store probability as an attribute
    if node.attrs.contains_key("ratio") {
        let prob = node.attrs.get("ratio").unwrap().clone().into_f32();
        return DropoutConfig::new(prob as f64);
    }

    if node.inputs.len() < 2 {
        panic!("Dropout configuration must have at least 2 inputs");
    }

    let ratio = node.inputs[1]
        .into_value()
        .clone()
        .expect("Dropout ratio must be passed in the second input")
        .data
        .into_scalar();

    let prob = match ratio {
        Data::Float16(ratio) => f64::from(f32::from(ratio)),
        Data::Float32(ratio) => ratio as f64,
        Data::Float64(ratio) => ratio,
        _ => panic!("Dropout ratio must be a float"),
    };

    DropoutConfig::new(prob)
}

pub struct DropoutProcessor;

impl NodeProcessor for DropoutProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (7, None)
    }

    fn process_config(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        graph_data: &mut crate::from_onnx::GraphData,
    ) {
        let config = dropout_config(node, graph_data);
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

    fn create_test_node_with_attr(ratio: f32) -> NodeBuilder {
        NodeBuilder::new(NodeType::Dropout, "test_dropout")
            .input_tensor_f32("data", 3, None)
            .output_tensor_f32("output", 3, None)
            .attr_float("ratio", ratio)
    }

    fn create_test_node_with_input(ratio: f32) -> NodeBuilder {
        NodeBuilder::new(NodeType::Dropout, "test_dropout")
            .input_tensor_f32("data", 3, None)
            .input_scalar_tensor_f32("ratio", Some(ratio))
            .output_tensor_f32("output", 3, None)
    }

    #[test]
    fn test_dropout_config_with_attr() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node_with_attr(0.3).build_with_graph_data(&mut graph_data);
        let config = dropout_config(&node, &mut graph_data);
        assert!(f64::abs(config.prob - 0.3) < 1e-6);
    }

    #[test]
    fn test_dropout_config_with_input() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node_with_input(0.5).build_with_graph_data(&mut graph_data);
        let config = dropout_config(&node, &mut graph_data);
        assert!(f64::abs(config.prob - 0.5) < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Dropout configuration must have at least 2 inputs")]
    fn test_dropout_config_missing_input() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let mut node = create_test_node_with_input(0.5).build_with_graph_data(&mut graph_data);
        node.attrs.clear(); // Remove attributes
        node.inputs.remove(1); // Remove ratio input
        let _ = dropout_config(&node, &mut graph_data);
    }
}
