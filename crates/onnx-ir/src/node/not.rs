//! # Logical NOT Operation
//!
//! Element-wise logical NOT operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Not.html>
//!
//! ## Type Constraints
//!
//! T: Boolean tensor types
//!
//! ## Opset Versions
//! - **Opset 1+**: Initial version

use onnx_ir_derive::NodeBuilderDerive;

use crate::ir::{Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
};

/// Node representation for Not operation
#[derive(Debug, Clone, NodeBuilderDerive)]
pub struct NotNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for logical NOT operation
pub(crate) struct NotProcessor;

impl NodeProcessor for NotProcessor {
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
        same_as_input(node);
        Ok(())
    }

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        Node::Not(NotNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
