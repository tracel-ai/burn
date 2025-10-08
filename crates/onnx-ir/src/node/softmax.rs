use crate::ir::{ArgType, Node};
use crate::processor::{NodeProcessor, ProcessorContext};

/// Create softmax config from the attributes of the node
pub fn softmax_config(node: &Node) -> usize {
    // the axis is the last dimension (Default: 1 per ONNX spec)
    let mut axis: i64 = -1;

    // check if the node has only one input
    if node.inputs.len() != 1 {
        panic!(
            "Softmax: multiple inputs are not supported (got {:?})",
            node.inputs.len()
        );
    }

    // extract the shape of the input tensor
    let tensor = match node.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // extract the attributes
    for (key, value) in node.attrs.iter() {
        if key.as_str() == "axis" {
            axis = value.clone().into_i64()
        }
    }

    // if axis is negative, it is counted from the end
    if axis < 0 {
        axis += tensor.rank as i64;
    }

    axis as usize
}

pub struct SoftmaxProcessor;

impl NodeProcessor for SoftmaxProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (1, None)
    }

    fn infer_outputs(&self, node: &mut Node, _context: &ProcessorContext) {
        crate::util::same_as_input(node);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(axis: i64, input_rank: usize) -> Node {
        NodeBuilder::new(NodeType::Softmax, "test_softmax")
            .input_tensor_f32("data", input_rank, None)
            .output_tensor_f32("output", input_rank, None)
            .attr_int("axis", axis)
            .build()
    }

    #[test]
    fn test_softmax_config_basic() {
        let node = create_test_node(-1, 3);
        let config = softmax_config(&node);
        assert_eq!(config, 2); // -1 + 3 = 2 (last dimension)
    }

    #[test]
    fn test_softmax_config_explicit_axis() {
        let node = create_test_node(1, 3);
        let config = softmax_config(&node);
        assert_eq!(config, 1);
    }

    #[test]
    #[should_panic(expected = "Softmax: multiple inputs are not supported")]
    fn test_softmax_config_multiple_inputs() {
        let mut node = create_test_node(1, 3);
        // Add an extra input
        let extra_input = NodeBuilder::new(NodeType::Identity, "temp")
            .input_tensor_f32("extra", 1, None)
            .build()
            .inputs
            .pop()
            .unwrap();
        node.inputs.push(extra_input);
        let _ = softmax_config(&node);
    }
}
