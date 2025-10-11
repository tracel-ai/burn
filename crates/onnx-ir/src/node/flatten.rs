use crate::ir::{ArgType, Node, NodeConfig, TensorType};
use crate::processor::{NodeProcessor, ProcessorContext};
use std::any::Any;

/// Configuration for Flatten operations
#[derive(Debug, Clone)]
pub struct FlattenConfig {
    /// Axis along which to flatten
    pub axis: usize,
}

impl NodeConfig for FlattenConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

/// Create a FlattenConfig from the attributes of the node
pub fn flatten_config(curr: &Node, _graph_data: &mut crate::from_onnx::GraphData) -> FlattenConfig {
    // the begin dimension is the first dimension (Default: 1 per ONNX spec)
    let mut axis: i64 = 1;

    // check if the node has only one input
    if curr.inputs.len() != 1 {
        panic!(
            "Flatten: multiple inputs are not supported (got {:?})",
            curr.inputs.len()
        );
    }

    // extract the shape of the input tensor
    let tensor = match curr.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // check if the input tensor has at least 2 dimensions
    if tensor.rank < 2 {
        panic!(
            "Flatten: input tensor must have at least 2 dimensions (got {:?})",
            tensor.rank
        );
    }

    // extract the attributes
    for (key, value) in curr.attrs.iter() {
        if key.as_str() == "axis" {
            axis = value.clone().into_i64()
        }
    }

    // if beg_dim is negative, it is counted from the end
    if axis < 0 {
        axis += tensor.rank as i64;
    }

    FlattenConfig {
        axis: axis as usize,
    }
}

pub struct FlattenProcessor;

impl NodeProcessor for FlattenProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (1, None)
    }

    fn process_config(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        graph_data: &mut crate::from_onnx::GraphData,
    ) {
        let config = flatten_config(node, graph_data);
        node.config = Some(Box::new(config));
    }

    fn process_forward(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        _graph_data: &mut crate::from_onnx::GraphData,
    ) {
        if node.inputs.len() != 1 {
            panic!("Flatten: multiple inputs are not supported");
        }
        let tensor = node
            .inputs
            .iter()
            .find_map(|input| match &input.ty {
                ArgType::Tensor(tensor) => Some(tensor),
                _ => None,
            })
            .unwrap();

        // Flatten to a 2D tensor
        node.outputs[0].ty = ArgType::Tensor(TensorType {
            rank: 2,
            ..tensor.clone()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axis: i64) -> Node {
        NodeBuilder::new(NodeType::Flatten, "test_flatten")
            .input_tensor_f32("data", 4, None)
            .output_tensor_f32("output", 2, None)
            .attr_int("axis", axis)
            .build()
    }

    #[test]
    fn test_flatten_config_basic() {
        let node = create_test_node(1);
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let config = flatten_config(&node, &mut graph_data);
        assert_eq!(config.axis, 1);
    }

    #[test]
    fn test_flatten_config_with_negative_axis() {
        let node = create_test_node(-2);
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let config = flatten_config(&node, &mut graph_data);
        assert_eq!(config.axis, 2); // -2 + 4 = 2
    }

    #[test]
    #[should_panic(expected = "Flatten: input tensor must have at least 2 dimensions")]
    fn test_flatten_config_with_low_rank() {
        let mut node = create_test_node(1);
        // Replace the input with one that has lower rank
        let input = NodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_f32("x", 1, None)
            .build()
            .inputs
            .pop()
            .unwrap();
        node.inputs[0] = input;
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let _ = flatten_config(&node, &mut graph_data);
    }

    #[test]
    #[should_panic(expected = "Flatten: multiple inputs are not supported")]
    fn test_flatten_config_with_multiple_inputs() {
        let mut node = create_test_node(1);
        // Add an extra input
        let extra_input = NodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_f32("extra", 1, None)
            .build()
            .inputs
            .pop()
            .unwrap();
        node.inputs.push(extra_input);
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let _ = flatten_config(&node, &mut graph_data);
    }
}
