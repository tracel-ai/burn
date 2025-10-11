use crate::ir::{ArgType, ElementType, Node};
use crate::processor::{NodeProcessor, ProcessorContext};

pub struct SizeProcessor;

impl NodeProcessor for SizeProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (1, None)
    }

    fn process_forward(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        _graph_data: &mut crate::from_onnx::GraphData,
    ) {
        log::debug!("Size rank inference for node {}", node.name);

        assert_eq!(
            node.inputs.len(),
            1,
            "Size: expected 1 input, found {}",
            node.inputs.len()
        );

        node.outputs[0].ty = ArgType::Scalar(ElementType::Int64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(rank: usize) -> Node {
        let builder = NodeBuilder::new(NodeType::Size, "test_size")
            .input_tensor_f32("data", rank, None)
            .output_scalar_i64("size");

        builder.build()
    }

    #[test]
    fn test_size_update_outputs() {
        let mut node = create_test_node(4);

        let processor = SizeProcessor;
        let context = ProcessorContext::new(16);
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        processor.process_forward(&mut node, &context, &mut graph_data);

        assert!(matches!(
            &node.outputs[0].ty,
            ArgType::Scalar(ElementType::Int64)
        ));
    }
}
