//! Max operation - element-wise maximum
//!
//! **ONNX Spec**: Opset 1+
//!
//! The Max operation computes element-wise maximum of two or more tensors.
//! Supports standard ONNX broadcasting semantics.

use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
    same_as_input_broadcast,
};

/// Node representation for Max operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct MaxNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for Max operation
pub(crate) struct MaxProcessor;

impl NodeProcessor for MaxProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::Exact(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Max supports variadic inputs (2+ inputs) per ONNX spec, not just 2
        same_as_input_broadcast(node);
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::Max(MaxNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
