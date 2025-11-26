//! # Bitwise NOT Operation
//!
//! Element-wise bitwise NOT operation.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__BitwiseNot.html>
//!
//! ## Type Constraints
//!
//! T: Integer tensor types (uint8, uint16, uint32, uint64, int8, int16, int32, int64)
//!
//! ## Opset Versions
//! - **Opset 18+**: Initial version

use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError, same_as_input,
    validate_opset,
};

/// Node representation for BitwiseNot operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct BitwiseNotNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for bitwise NOT operation
pub(crate) struct BitwiseNotProcessor;

impl NodeProcessor for BitwiseNotProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 18,
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
        validate_opset(opset, 18)?;
        same_as_input(node);
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::BitwiseNot(BitwiseNotNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
