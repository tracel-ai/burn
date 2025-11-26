//! # Logical XOR Operation
//!
//! Element-wise logical XOR operation with multidirectional broadcasting support.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Xor.html>
//!
//! ## Type Constraints
//!
//! T: Boolean tensor types
//!
//! ## Opset Versions
//! - **Opset 1-6**: Limited broadcast support
//! - **Opset 7+**: Multidirectional (Numpy-style) broadcasting

use onnx_ir_derive::NodeBuilder;

use crate::ir::{Argument, Node, RawNode};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
    same_as_input_broadcast,
};

/// Node representation for Xor operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct XorNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for logical XOR operation
pub(crate) struct XorProcessor;

impl NodeProcessor for XorProcessor {
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
        same_as_input_broadcast(node);
        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::Xor(XorNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
