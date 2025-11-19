//! # Floor Operation
//!
//! Element-wise floor operation (round down to nearest integer).
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Floor.html>
//!
//! ## Type Constraints
//!
//! T: Float tensor types
//!
//! ## Opset Versions
//! - **Opset 1-5**: Basic support
//! - **Opset 6+**: Improved shape inference
//! - **Opset 13+**: Added support for bfloat16

use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
    validate_opset,
};

/// Node representation for Floor operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct FloorNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for floor operation
pub(crate) struct FloorProcessor;

impl NodeProcessor for FloorProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 6,
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
        validate_opset(opset, 6)?;
        same_as_input(node);
        Ok(())
    }

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        Node::Floor(FloorNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
