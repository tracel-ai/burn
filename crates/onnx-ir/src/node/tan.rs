//! # Tangent Operation
//!
//! Element-wise tangent operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Tan.html>
//!
//! ## Type Constraints
//!
//! T: Float tensor types
//!
//! ## Opset Versions
//! - **Opset 7+**: Initial version

use onnx_ir_derive::NodeBuilderDerive;

use crate::ir::{Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
    validate_opset,
};

/// Node representation for Tan operation
#[derive(Debug, Clone, NodeBuilderDerive)]
pub struct TanNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for tangent operation
pub(crate) struct TanProcessor;

impl NodeProcessor for TanProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 7,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut NodeBuilder,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        validate_opset(opset, 7)?;
        same_as_input(node);
        Ok(())
    }

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        Node::Tan(TanNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
