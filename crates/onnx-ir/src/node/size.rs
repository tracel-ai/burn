use crate::ir::{ArgType, ElementType, Node};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};

pub struct SizeProcessor;

impl NodeProcessor for SizeProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::processor::validate_opset(opset, 1)?;
        crate::processor::validate_input_count(node, 1)?;
        crate::processor::validate_output_count(node, 1)?;

        log::debug!("Size rank inference for node {}", node.name);

        node.outputs[0].ty = ArgType::Scalar(ElementType::Int64);

        Ok(())
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
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(matches!(
            &node.outputs[0].ty,
            ArgType::Scalar(ElementType::Int64)
        ));
    }
}
