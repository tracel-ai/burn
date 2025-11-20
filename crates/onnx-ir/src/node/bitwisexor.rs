//! # Bitwise XOR Operation
//!
//! Element-wise bitwise XOR operation with multidirectional broadcasting support.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__BitwiseXor.html>
//!
//! ## Type Constraints
//!
//! T: Integer tensor types (int8, int16, int32, int64, uint8, uint16, uint32, uint64)
//!
//! ## Opset Versions
//! - **Opset 18+**: Bitwise operations introduced

use onnx_ir_derive::NodeBuilderDerive;

use crate::ir::{Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
    same_as_input_broadcast,
};

/// Node representation for BitwiseXor operation
#[derive(Debug, Clone, NodeBuilderDerive)]
pub struct BitwiseXorNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for bitwise XOR operation
pub(crate) struct BitwiseXorProcessor;

impl NodeProcessor for BitwiseXorProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 18,
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
        same_as_input_broadcast(node);
        Ok(())
    }

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        Node::BitwiseXor(BitwiseXorNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
