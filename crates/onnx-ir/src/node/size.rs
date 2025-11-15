//! # Size
//!
//! Returns the total number of elements in the input tensor as a scalar int64 value.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Size.html>
//!
//! ## Opset Versions
//! - **Since version 23**: Current version
//! - **Since version 1**: Initial implementation

use crate::ir::{ArgType, DType, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

pub(crate) struct SizeProcessor;

impl NodeProcessor for SizeProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        node.outputs[0].ty = ArgType::Scalar(DType::I64);

        Ok(())
    }

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        Node::Size {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(rank: usize) -> NodeBuilder {
        let builder = TestNodeBuilder::new(NodeType::Size, "test_size")
            .input_tensor_f32("data", rank, None)
            .output_scalar_i64("size");

        builder.build()
    }

    // TODO: Missing test for zero-size tensors - what is the size of a tensor with shape [0, 5, 3]?
    // Should be 0 (0 * 5 * 3 = 0) but this edge case is not tested.

    // TODO: Missing test for scalar (rank-0) tensors - what is the size of a scalar?
    // Should be 1 according to ONNX spec but not validated.

    // TODO: Missing test for very large tensors - size can overflow i64 for extremely large tensors.
    // Need to verify behavior when product of dimensions exceeds i64::MAX.

    #[test]
    fn test_size_update_outputs() {
        let mut node = create_test_node(4);

        let processor = SizeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        assert!(matches!(&node.outputs[0].ty, ArgType::Scalar(DType::I64)));
    }
}
