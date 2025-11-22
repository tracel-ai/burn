//! # Power Operation
//!
//! Element-wise power operation (a^b) with multidirectional broadcasting support.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Pow.html>
//!
//! ## Type Constraints
//!
//! T: Numeric tensor types (float16, float32, float64, int32, int64)
//!
//! ## Opset Versions
//! - **Opset 1-6**: Limited broadcast support
//! - **Opset 7-11**: Multidirectional broadcasting, added type support
//! - **Opset 12-14**: Extended type support (bfloat16)
//! - **Opset 15+**: Extended integer type support

use onnx_ir_derive::NodeBuilderDerive;

use crate::ir::{Argument, Node, NodeBuilder};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
    same_as_input_broadcast,
};

/// Node representation for Pow operation
#[derive(Debug, Clone, NodeBuilderDerive)]
pub struct PowNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

/// Node processor for power operation
pub(crate) struct PowProcessor;

impl NodeProcessor for PowProcessor {
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
        same_as_input_broadcast(node);
        Ok(())
    }

    fn build_node(&self, builder: NodeBuilder, _opset: usize) -> Node {
        Node::Pow(PowNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}
