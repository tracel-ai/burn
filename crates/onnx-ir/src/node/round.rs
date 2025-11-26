//! # Round Operation
//!
//! Element-wise rounding operation (round to nearest integer).
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Round.html>
//!
//! ## Type Constraints
//!
//! T: Float tensor types
//!
//! ## Opset Versions
//! - **Opset 11+**: Initial version with rounding mode support

use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
    validate_opset,
};

/// Node representation for Round operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct RoundNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for round operation
pub(crate) struct RoundProcessor;

impl NodeProcessor for RoundProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 11,
            max_opset: None,
            inputs: InputSpec::Exact(1),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        validate_opset(opset, 11)?;
        same_as_input(node);
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::Round(RoundNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
