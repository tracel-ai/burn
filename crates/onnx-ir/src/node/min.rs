//! Min operation - element-wise minimum
//!
//! **ONNX Spec**: Opset 1+
//!
//! The Min operation computes element-wise minimum of two or more tensors.
//! Supports standard ONNX broadcasting semantics.

use onnx_ir_derive::NodeBuilderDerive;

use crate::ir::{Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
    same_as_input_broadcast,
};

/// Node representation for Min operation
#[derive(Debug, Clone, NodeBuilderDerive)]
pub struct MinNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for Min operation
pub(crate) struct MinProcessor;

impl NodeProcessor for MinProcessor {
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
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Min supports variadic inputs (2+ inputs) per ONNX spec, not just 2
        same_as_input_broadcast(node);
        Ok(())
    }

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        Node::Min(MinNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
